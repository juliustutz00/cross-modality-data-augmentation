import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu, threshold_li
import cv2
from ..enums import Input_Modality, Output_Modality


def transform_color(image_to_be_transformed: np.ndarray, reference: np.ndarray, ratio: float, input_modality: Input_Modality, output_modality: Output_Modality):
    """
    Changes color of an image to that of the reference image by adjusting the 
    image so that a fraction of its cumulative histogram matches that of another.

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    reference : np.ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    input_modality : Input_Modality
        The modality of the image_to_be_transformed. If Input_Modality.any is 
        chosen, there are no modality-specific adjustments.
    output_modality : Output_Modality
        The desired modality of the transformation. If Output_Modality.
        custom is chosen, custom reference images can be chosen.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the given ration is not between 0 and 1.
        Thrown when input and reference image do not have the same shape.
        Thrown when input modality is not implemented.
        Thrown when output modality is not implemented.
    """
    from skimage.exposure import match_histograms

    if ratio < 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1.")

    # Ensure both images have the same number of channels
    if image_to_be_transformed.ndim != reference.ndim:
        raise ValueError("Number of channels in the input image and reference image must match!")
    
    if len(image_to_be_transformed.shape) == 2:
        # Grayscale image
        # Identify non-black pixels
        non_black_pixels = image_to_be_transformed > 3
        reference_non_black_pixels = reference > 7

        # Perform histogram matching only on non-black pixels
        matched_non_black_pixels = match_histograms(
            image_to_be_transformed[non_black_pixels],
            reference[reference_non_black_pixels]
        )
    else:
        # Color image
        non_black_pixels = np.all(image_to_be_transformed > [15, 15, 15], axis=-1)
        reference_non_black_pixels = np.all(reference > [7, 7, 7], axis=-1)

        transformed_pixels = image_to_be_transformed[non_black_pixels]
        reference_pixels = reference[reference_non_black_pixels]

        matched_pixels = match_histograms(transformed_pixels, reference_pixels, channel_axis=-1)
    

    if len(image_to_be_transformed.shape) == 2:
        match input_modality:
            case Input_Modality.any:
                match output_modality:
                    case Output_Modality.PET:
                        return refine_image_details_any_to_PET(image_to_be_transformed, ratio, matched_non_black_pixels, non_black_pixels)
                    case Output_Modality.MRI:
                        pass
                    case Output_Modality.CT:
                        pass
                    case Output_Modality.custom:
                        pass
                    case _:
                        raise ValueError("Output modality is not implemented.")
            case Input_Modality.PET:
                match output_modality:
                    case Output_Modality.PET:
                        pass
                    case Output_Modality.MRI:
                        return refine_image_details_PET_to_MRI(image_to_be_transformed, ratio, matched_non_black_pixels, non_black_pixels)
                    case Output_Modality.CT:
                        return refine_image_details_PET_to_CT(image_to_be_transformed, ratio, matched_non_black_pixels, non_black_pixels)
                    case Output_Modality.custom:
                        pass
                    case _:
                        raise ValueError("Output modality is not implemented.")
            case Input_Modality.MRI:
                match output_modality:
                    case Output_Modality.PET:
                        return refine_image_details_MRI_to_PET(image_to_be_transformed, ratio, matched_non_black_pixels, non_black_pixels)
                    case Output_Modality.MRI:
                        pass
                    case Output_Modality.CT:
                        return refine_image_details_MRI_to_CT(image_to_be_transformed, ratio, matched_non_black_pixels, non_black_pixels)
                    case Output_Modality.custom:
                        pass
                    case _:
                        raise ValueError("Output modality is not implemented.")
            case Input_Modality.CT:
                match output_modality:
                    case Output_Modality.PET:
                        return refine_image_details_CT_to_PET(image_to_be_transformed, ratio, matched_non_black_pixels, non_black_pixels)
                    case Output_Modality.MRI:
                        return refine_image_details_CT_to_MRI(image_to_be_transformed, ratio, matched_non_black_pixels, non_black_pixels)
                    case Output_Modality.CT:
                        pass
                    case Output_Modality.custom:
                        pass
                    case _:
                        raise ValueError("Output modality is not implemented.")
            case _:
                raise ValueError("Input modality is not implemented.")
        
    # Mix the adjusted and original images based on the ratio
    mixed_image = np.copy(image_to_be_transformed)
    
    if len(image_to_be_transformed.shape) == 2:
        mixed_image[non_black_pixels] = np.round(
            (1 - ratio) * mixed_image[non_black_pixels] + ratio * matched_non_black_pixels
        ).astype(np.uint8)
    else: 
        mixed_image[non_black_pixels] = np.round(
            (1 - ratio) * transformed_pixels + ratio * matched_pixels
        ).astype(np.uint8)

    return mixed_image



def refine_image_details_any_to_PET(image_to_be_transformed: np.ndarray, ratio: float, matched_non_black_pixels: np.ndarray, non_black_pixels: np.ndarray):
    """
    Refines certain PET image details from an image from any modality. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    matched_non_black_pixels : np.ndarray
        Result of histogram matching applied to non-black pixels in the input 
        image, adjusted to match the histogram of the reference image's 
        non-black pixels.
    non_black_pixels : np.ndarray
        Mask representing non-black pixels in the input image, typically used
        to exclude black areas from the histogram matching process.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.
    """
    # Calculate the intensity value corresponding to the 90th and 25th percentile of the input image
    percentile_90 = np.percentile(image_to_be_transformed, 90)
    percentile_25 = np.percentile(image_to_be_transformed, 25)

    # Mask for pixels to be adjusted (skull, white pixels)
    mask_above = image_to_be_transformed < percentile_90
    mask_above = mask_above > percentile_25
    mask_above_2 = image_to_be_transformed >= percentile_90

    # Amplify the color of specific regions (light up brain, darken skull)
    matched_non_black_pixels[mask_above[non_black_pixels]] = np.round(matched_non_black_pixels[mask_above[non_black_pixels]] * 3.0) 
    matched_non_black_pixels[mask_above_2[non_black_pixels]] = np.round(matched_non_black_pixels[mask_above_2[non_black_pixels]] * 0.1) 
    matched_non_black_pixels = np.clip(matched_non_black_pixels, 0, 255).astype(np.uint8)

    # Mix the adjusted and original images based on the ratio
    mixed_image = np.copy(image_to_be_transformed)
    mixed_image[non_black_pixels] = np.round(
        (1 - ratio) * mixed_image[non_black_pixels] + ratio * matched_non_black_pixels
    ).astype(np.uint8)

    return mixed_image


def refine_image_details_PET_to_MRI(image_to_be_transformed: np.ndarray, ratio: float, matched_non_black_pixels: np.ndarray, non_black_pixels: np.ndarray):
    """
    Refines certain MRI image details from a PET image. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    matched_non_black_pixels : np.ndarray
        Result of histogram matching applied to non-black pixels in the input 
        image, adjusted to match the histogram of the reference image's 
        non-black pixels.
    non_black_pixels : np.ndarray
        Mask representing non-black pixels in the input image, typically used
        to exclude black areas from the histogram matching process.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.
    """
    # Mix the adjusted and original images based on the ratio
    mixed_image = np.copy(image_to_be_transformed)
    mixed_image[non_black_pixels] = np.round(
        (1 - ratio) * mixed_image[non_black_pixels] + ratio * matched_non_black_pixels
    ).astype(np.uint8)

    # Paint skull white, brighten dark spots
    # Threshold to create skull mask
    threshold = threshold_li(image_to_be_transformed)
    skull_mask = image_to_be_transformed > threshold
    skull_mask_filled = binary_fill_holes(skull_mask)

    # Compute distance transform
    distance = cv2.distanceTransform(skull_mask_filled.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inverted_distance = np.max(distance) - distance

    # Define thresholds for outer regions
    outer_5_percent_threshold = np.percentile(inverted_distance[skull_mask_filled == 1], 95)
    outer_20_percent_threshold = np.percentile(inverted_distance[skull_mask_filled == 1], 80)

    # Create masks for outer regions
    outer_5_percent_mask = (inverted_distance >= outer_5_percent_threshold) & (skull_mask_filled == 1)
    outer_20_percent_mask = (inverted_distance < outer_5_percent_threshold) & (inverted_distance >= outer_20_percent_threshold) & (skull_mask_filled == 1)

    # Smooth and blend outer 20 percent region
    smooth_region_outer_20 = outer_20_percent_mask
    smoothed_region_outer_20 = cv2.GaussianBlur(smooth_region_outer_20.astype(np.float32), (3, 3), 0)
    smoothed_region_outer_20 = ratio * (smoothed_region_outer_20 / np.max(smoothed_region_outer_20))
    mixed_image = mixed_image * (1 - smoothed_region_outer_20) + 0 * smoothed_region_outer_20
    mixed_image = mixed_image.astype(np.uint8)

    # Smooth and blend outer 5 percent region
    smoothed_region_outer_5 = cv2.GaussianBlur(outer_5_percent_mask.astype(np.float32), (3, 3), 0)
    smoothed_region_outer_5 = ratio * (smoothed_region_outer_5 / np.max(smoothed_region_outer_5))
    mixed_image = mixed_image * (1 - smoothed_region_outer_5) + 150 * smoothed_region_outer_5
    mixed_image = mixed_image.astype(np.uint8)

    # Define inner pixel mask and dark pixel values
    inner_pixel_mask = (inverted_distance < outer_20_percent_threshold) & (skull_mask_filled == 1)
    dark_pixel_values = mixed_image[inner_pixel_mask]
    dark_pixel_threshold_20 = np.percentile(dark_pixel_values, 20)

    # Create and smooth darkest 20 percent mask
    darkest_20_percent_mask = inner_pixel_mask & (mixed_image <= dark_pixel_threshold_20)
    smoothed_region_darkest_20 = cv2.GaussianBlur(darkest_20_percent_mask.astype(np.float32), (3, 3), 0)
    smoothed_region_darkest_20 = ratio * (smoothed_region_darkest_20 / np.max(smoothed_region_darkest_20))
    mixed_image = mixed_image * (1 - smoothed_region_darkest_20) + 180 * smoothed_region_darkest_20
    mixed_image = mixed_image.astype(np.uint8)

    return mixed_image

def refine_image_details_PET_to_CT(image_to_be_transformed: np.ndarray, ratio: float, matched_non_black_pixels: np.ndarray, non_black_pixels: np.ndarray):
    """
    Refines certain CT image details from a PET image. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    matched_non_black_pixels : np.ndarray
        Result of histogram matching applied to non-black pixels in the input 
        image, adjusted to match the histogram of the reference image's 
        non-black pixels.
    non_black_pixels : np.ndarray
        Mask representing non-black pixels in the input image, typically used
        to exclude black areas from the histogram matching process.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.
    """
    # Mix the adjusted and original images based on the ratio
    mixed_image = np.copy(image_to_be_transformed)
    mixed_image[non_black_pixels] = np.round(
        (1 - ratio) * mixed_image[non_black_pixels] + ratio * matched_non_black_pixels
    ).astype(np.uint8)

    # Paint skull white, brighten dark spots
    # Threshold to create skull mask
    threshold = threshold_li(image_to_be_transformed)
    skull_mask = image_to_be_transformed > threshold
    skull_mask_filled = binary_fill_holes(skull_mask)

    # Compute distance transform
    distance = cv2.distanceTransform(skull_mask_filled.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inverted_distance = np.max(distance) - distance

    # Define thresholds and create masks for outer regions
    try:
        threshold = np.percentile(inverted_distance[skull_mask_filled == 1], 85)
    except IndexError:
        threshold = 5.0
    outer_15_percent_mask = (inverted_distance >= threshold) & (skull_mask_filled == 1)
    darkest_spots_mask = (image_to_be_transformed < 50) & (skull_mask_filled == 1)

    # Brighten outer regions
    smoothed_region = cv2.GaussianBlur(outer_15_percent_mask.astype(np.float32), (3, 3), 0)
    smoothed_region = ratio * (smoothed_region / np.max(smoothed_region))
    adjusted_brightness = mixed_image[darkest_spots_mask] * (2 * ratio)
    mixed_image[darkest_spots_mask] = np.clip(adjusted_brightness, 0, 255).astype(np.uint8)
    mixed_image = mixed_image * (1 - smoothed_region) + 255 * smoothed_region
    mixed_image = mixed_image.astype(np.uint8)

    return mixed_image


def refine_image_details_MRI_to_PET(image_to_be_transformed: np.ndarray, ratio: float, matched_non_black_pixels: np.ndarray, non_black_pixels: np.ndarray):
    """
    Refines certain PET image details from a MRI image. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    matched_non_black_pixels : np.ndarray
        Result of histogram matching applied to non-black pixels in the input 
        image, adjusted to match the histogram of the reference image's 
        non-black pixels.
    non_black_pixels : np.ndarray
        Mask representing non-black pixels in the input image, typically used
        to exclude black areas from the histogram matching process.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.
    """
    # Calculate the intensity value corresponding to the 90th and 25th percentile of the input image
    percentile_90 = np.percentile(image_to_be_transformed, 90)
    percentile_25 = np.percentile(image_to_be_transformed, 25)

    # Mask for pixels to be adjusted (skull, white pixels)
    mask_above = image_to_be_transformed < percentile_90
    mask_above = mask_above > percentile_25
    mask_above_2 = image_to_be_transformed >= percentile_90

    # Amplify the color of specific regions (light up brain, darken skull)
    matched_non_black_pixels[mask_above[non_black_pixels]] = np.round(matched_non_black_pixels[mask_above[non_black_pixels]] * 3.0) 
    matched_non_black_pixels[mask_above_2[non_black_pixels]] = np.round(matched_non_black_pixels[mask_above_2[non_black_pixels]] * 0.1) 
    matched_non_black_pixels = np.clip(matched_non_black_pixels, 0, 255).astype(np.uint8)

    # Mix the adjusted and original images based on the ratio
    mixed_image = np.copy(image_to_be_transformed)
    mixed_image[non_black_pixels] = np.round(
        (1 - ratio) * mixed_image[non_black_pixels] + ratio * matched_non_black_pixels
    ).astype(np.uint8)

    return mixed_image

def refine_image_details_MRI_to_CT(image_to_be_transformed: np.ndarray, ratio: float, matched_non_black_pixels: np.ndarray, non_black_pixels: np.ndarray):
    """
    Refines certain CT image details from a MRI image. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    matched_non_black_pixels : np.ndarray
        Result of histogram matching applied to non-black pixels in the input 
        image, adjusted to match the histogram of the reference image's 
        non-black pixels.
    non_black_pixels : np.ndarray
        Mask representing non-black pixels in the input image, typically used
        to exclude black areas from the histogram matching process.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.
    """
    # Mix the adjusted and original images based on the ratio
    mixed_image = np.copy(image_to_be_transformed)
    mixed_image[non_black_pixels] = np.round(
        (1 - ratio) * mixed_image[non_black_pixels] + ratio * matched_non_black_pixels
    ).astype(np.uint8)

    # Paint skull white
    # Threshold to create skull mask
    threshold = threshold_otsu(image_to_be_transformed)
    skull_mask = image_to_be_transformed > threshold
    skull_mask_filled = binary_fill_holes(skull_mask)

    # Compute distance transform
    distance = cv2.distanceTransform(skull_mask_filled.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inverted_distance = np.max(distance) - distance

    # Define thresholds and create masks for outer regions
    try:
        threshold = np.percentile(inverted_distance[skull_mask_filled == 1], 85)
    except IndexError:
        threshold = 50.0
    outer_15_percent_mask = (inverted_distance >= threshold) & (skull_mask_filled == 1)

    # Brighten outer regions
    smoothed_region = cv2.GaussianBlur(outer_15_percent_mask.astype(np.float32), (3, 3), 0)
    smoothed_region = ratio * (smoothed_region / np.max(smoothed_region))
    mixed_image = mixed_image * (1 - smoothed_region) + 255 * smoothed_region
    mixed_image = mixed_image.astype(np.uint8)

    return mixed_image


def refine_image_details_CT_to_PET(image_to_be_transformed: np.ndarray, ratio: float, matched_non_black_pixels: np.ndarray, non_black_pixels: np.ndarray):
    """
    Refines certain PET image details from a CT image. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    matched_non_black_pixels : np.ndarray
        Result of histogram matching applied to non-black pixels in the input 
        image, adjusted to match the histogram of the reference image's 
        non-black pixels.
    non_black_pixels : np.ndarray
        Mask representing non-black pixels in the input image, typically used
        to exclude black areas from the histogram matching process.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.
    """
    # Calculate the intensity value corresponding to the 90th and 25th percentile of the input image
    percentile_90 = np.percentile(image_to_be_transformed, 90)
    percentile_25 = np.percentile(image_to_be_transformed, 25)

    # Mask for pixels to be adjusted (skull, white pixels)
    mask_above = image_to_be_transformed < percentile_90
    mask_above = mask_above > percentile_25
    mask_above_2 = image_to_be_transformed >= percentile_90

    # Amplify the color of specific regions (light up brain, darken skull)
    matched_non_black_pixels[mask_above[non_black_pixels]] = np.round(matched_non_black_pixels[mask_above[non_black_pixels]] * 3.0) 
    matched_non_black_pixels[mask_above_2[non_black_pixels]] = np.round(matched_non_black_pixels[mask_above_2[non_black_pixels]] * 0.1) 
    matched_non_black_pixels = np.clip(matched_non_black_pixels, 0, 255).astype(np.uint8)

    # Mix the adjusted and original images based on the ratio
    mixed_image = np.copy(image_to_be_transformed)
    mixed_image[non_black_pixels] = np.round(
        (1 - ratio) * mixed_image[non_black_pixels] + ratio * matched_non_black_pixels
    ).astype(np.uint8)

    return mixed_image

def refine_image_details_CT_to_MRI(image_to_be_transformed: np.ndarray, ratio: float, matched_non_black_pixels: np.ndarray, non_black_pixels: np.ndarray):
    """
    Refines certain MRI image details from a CT image. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    matched_non_black_pixels : np.ndarray
        Result of histogram matching applied to non-black pixels in the input 
        image, adjusted to match the histogram of the reference image's 
        non-black pixels.
    non_black_pixels : np.ndarray
        Mask representing non-black pixels in the input image, typically used
        to exclude black areas from the histogram matching process.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.
    """
    # Mask for pixels to be adjusted (skull, white pixels)
    mask_above = image_to_be_transformed >= 255

    matched_non_black_pixels[mask_above[non_black_pixels]] = np.round(matched_non_black_pixels[mask_above[non_black_pixels]] * 1.4)
    matched_non_black_pixels = np.clip(matched_non_black_pixels, 0, 255).astype(np.uint8)

    # Mix the adjusted and original images based on the ratio
    mixed_image = np.copy(image_to_be_transformed)
    mixed_image[non_black_pixels] = np.round(
        (1 - ratio) * mixed_image[non_black_pixels] + ratio * matched_non_black_pixels
    ).astype(np.uint8)

    return mixed_image

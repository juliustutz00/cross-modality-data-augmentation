import numpy as np
import cv2
import random
from ..enums import Input_Modality, Output_Modality


def transform_artifact(image_to_be_transformed: np.ndarray, ratio: float, input_modality: Input_Modality, output_modality: Output_Modality): 
    '''
    Adds a modality characteristic artifact to a given image.
    Is only executed about every 10th image as artifacts are rather rare in reality.

    Knowledge about modality characteristic noise is taken from:
    - Abouzied, Mohei M., Elpida S. Crawford, and Hani Abdel Nabi. "18F-FDG imaging: pitfalls and artifacts." Journal of nuclear medicine technology 33.3 (2005): 145-155. 
    - Sureshbabu, Waheeda, and Osama Mawlawi. "PET/CT imaging artifacts." Journal of nuclear medicine technology 33.3 (2005): 156-161. 
    - Cook, Gary JR, Eva A. Wegner, and Ignac Fogelman. "Pitfalls and artifacts in 18FDG PET and PET/CT oncologic imaging." Seminars in nuclear medicine. Vol. 34. No. 2. WB Saunders, 2004. 
    - Krupa, Katarzyna, and Monika Bekiesińska-Figatowska. "Artifacts in magnetic resonance imaging." Polish journal of radiology 80 (2015): 93. 
    - Smith, Travis B. "MRI artifacts and correction strategies." Imaging in Medicine 2.4 (2010): 445. 
    - Boas, F. Edward, and Dominik Fleischmann. "CT artifacts: causes and reduction techniques." Imaging Med 4.2 (2012): 229-240. 
    - Barrett, Julia F., and Nicholas Keat. "Artifacts in CT: recognition and avoidance." Radiographics 24.6 (2004): 1679-1691. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    input_modality : Input_Modality
        The modality of the image_to_be_transformed. 
    output_modality : Output_Modality
        The desired modality of the transformation. If Output_Modality.custom 
        is chosen, there are no modality-specific adjustments.

    Returns
    -------
    mixed_image : np.ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the given ration is not between 0 and 1.
        Thrown when input modality is not implemented.
        Thrown when output modality is not implemented.
    '''
    if ratio < 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1.")

    match input_modality:
        case Input_Modality.any:
            match output_modality:
                case Output_Modality.PET:
                    return add_PET_artifact(image_to_be_transformed, ratio)
                case Output_Modality.MRI:
                    return add_MRI_artifact(image_to_be_transformed, ratio)
                case Output_Modality.CT:
                    return add_CT_artifact(image_to_be_transformed, ratio)
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case Input_Modality.PET:
            match output_modality:
                case Output_Modality.PET:
                    pass
                case Output_Modality.MRI:
                    return add_MRI_artifact(image_to_be_transformed, ratio)
                case Output_Modality.CT:
                    return add_CT_artifact(image_to_be_transformed, ratio)
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case Input_Modality.MRI:
            match output_modality:
                case Output_Modality.PET:
                    return add_PET_artifact(image_to_be_transformed, ratio)
                case Output_Modality.MRI:
                    pass
                case Output_Modality.CT:
                    return add_CT_artifact(image_to_be_transformed, ratio)
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case Input_Modality.CT:
            match output_modality:
                case Output_Modality.PET:
                    return add_PET_artifact(image_to_be_transformed, ratio)
                case Output_Modality.MRI:
                    return add_MRI_artifact(image_to_be_transformed, ratio)
                case Output_Modality.CT:
                    pass
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case _:
            raise ValueError("Input modality is not implemented.")
        
    # When no implemented modality transformation is chosen, the image is not altered
    return image_to_be_transformed



def add_PET_artifact(image, ratio): 
    """
    Adds a random PET characteristic artifact to an image.

    Knowledge about artifacts is taken from: 
    - Abouzied, Mohei M., Elpida S. Crawford, and Hani Abdel Nabi. "18F-FDG imaging: pitfalls and artifacts." Journal of nuclear medicine technology 33.3 (2005): 145-155. 
    - Sureshbabu, Waheeda, and Osama Mawlawi. "PET/CT imaging artifacts." Journal of nuclear medicine technology 33.3 (2005): 156-161. 
    - Cook, Gary JR, Eva A. Wegner, and Ignac Fogelman. "Pitfalls and artifacts in 18FDG PET and PET/CT oncologic imaging." Seminars in nuclear medicine. Vol. 34. No. 2. WB Saunders, 2004. 

    Parameters 
    ----------
    image : np.ndarray 
        The original image.
    ratio: 
        The ratio used to determine the intensity of the artifact.        
    
    Returns 
    ----------
    output_image : np.ndarray 
        The image with a random added artifact.
    """
    artifact_functions = [
        lambda img, rt: add_metal_object_artifact_PET_MRI(img, rt),
        lambda img, rt: add_motion_artifact_PET(img, rt),
        lambda img, rt: add_attenuation_artifact_PET(img, rt)
    ]

    selected_function = random.choice(artifact_functions)
    return selected_function(image, ratio)

def add_MRI_artifact(image, ratio): 
    """
    Adds a random MRI characteristic artifact to an image.

    Knowledge about artifacts is taken from: 
    - Krupa, Katarzyna, and Monika Bekiesińska-Figatowska. "Artifacts in magnetic resonance imaging." Polish journal of radiology 80 (2015): 93. 
    - Smith, Travis B. "MRI artifacts and correction strategies." Imaging in Medicine 2.4 (2010): 445. 
    
    Parameters 
    ----------
    image : np.ndarray 
        The original image.
    ratio: 
        The ratio used to determine the intensity of the artifact.        
    
    Returns 
    ----------
    output_image : np.ndarray 
        The image with a random added artifact.
    """
    artifact_functions = [
        lambda img, rt: add_metal_object_artifact_PET_MRI(img, rt),
        lambda img, rt: add_motion_artifact_MRI(img, rt),
        lambda img, rt: add_gibbs_artifact_MRI(img, rt),
        lambda img, rt: add_chemical_shift_artifact_MRI(img, rt)
    ]

    selected_function = random.choice(artifact_functions)
    return selected_function(image, ratio)

def add_CT_artifact(image, ratio): 
    """
    Adds a random CT characteristic artifact to an image.

    Knowledge about artifacts is taken from: 
    - Boas, F. Edward, and Dominik Fleischmann. "CT artifacts: causes and reduction techniques." Imaging Med 4.2 (2012): 229-240. 
    - Barrett, Julia F., and Nicholas Keat. "Artifacts in CT: recognition and avoidance." Radiographics 24.6 (2004): 1679-1691. 
    - Cook, Gary JR, Eva A. Wegner, and Ignac Fogelman. "Pitfalls and artifacts in 18FDG PET and PET/CT oncologic imaging." Seminars in nuclear medicine. Vol. 34. No. 2. WB Saunders, 2004. 
    
    Parameters 
    ----------
    image : np.ndarray 
        The original image.
    ratio: 
        The ratio used to determine the intensity of the artifact.        
    
    Returns 
    ----------
    output_image : np.ndarray 
        The image with a random added artifact.
    """
    artifact_functions = [
        lambda img, rt: add_metal_object_artifact_CT(img, rt),
        lambda img, rt: add_motion_artifact_CT(img, rt),
        lambda img, rt: add_beam_hardening_artifact_CT(img, rt),
        lambda img, rt: add_ring_artifact_CT(img, rt)
    ]

    selected_function = random.choice(artifact_functions)
    return selected_function(image, ratio)


def add_metal_object_artifact_PET_MRI(image, ratio):
    """
    Inserts a black spot with a smooth transition at a random location in the image.
    
    Parameters
    ----------
    image : np.ndarray 
        The original MRI or PET image. Can be grayscale or color.
    ratio : float 
        The scaling factor for the radius of the black spot.
    
    Returns
    ----------
    output_image : np.ndarray 
        The image with the inserted black spot.
    """
    output_image = image.copy()
    if len(image.shape) == 2:  # Grayscale image
        height, width = output_image.shape
    else:  # Color image
        height, width, channels = output_image.shape
    
    # Choose a random position for the black spot
    center_x = random.randint(0, width-1)
    center_y = random.randint(0, height-1)
    
    # Choose a random radius for the black spot
    radius = int(20 * ratio)
    
    # Create a circular mask with a smooth transition
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.circle(mask, (center_x, center_y), radius, (1.0,), thickness=-1)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)  # Smooth edges with Gaussian Blur
    
    # Normalize the mask to the range 0-1
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    
    # Invert the mask for the black spot
    inverted_mask = 1 - mask
    
    if len(image.shape) == 2:  # Grayscale image
        # Apply the mask to the grayscale image
        output_image = (output_image * inverted_mask).astype(np.uint8)
    else:  # Color image
        # Apply the mask to each channel independently
        for c in range(channels):
            output_image[..., c] = (output_image[..., c] * inverted_mask).astype(np.uint8)
    
    return output_image

def add_metal_object_artifact_CT(image, ratio):
    """
    Draws thin white rays emanating from a bright spot in all directions.
    
    Parameters
    ----------
    image : np.ndarray 
        The original CT image. Can be grayscale or color.
    ratio : float 
        A blending ratio between the original image and the image with rays.

    Returns
    ----------
    output_image : np.ndarray 
        The image with drawn rays.
    """
    num_directions = 45
    max_length = 150
    intensity_decrease = 0.97

    output_image = image.copy()
    
    if len(image.shape) == 2:  # Grayscale image
        height, width = output_image.shape
        channels = 1
    else:  # Color image
        height, width, channels = output_image.shape

    # Find all pixels with value 255 (bright spots) for grayscale or all channels bright for color
    if channels == 1:
        bright_spots = np.argwhere(output_image > 200)
    else:
        bright_spots = np.argwhere(np.all(output_image > [200, 200, 200], axis=-1))
    
    if bright_spots.size == 0:
        return output_image 
    
    # Randomly choose a bright spot
    chosen_spot = bright_spots[random.randint(0, len(bright_spots) - 1)]
    y, x = chosen_spot[0], chosen_spot[1]
    
    # Generate directions in angular steps
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    
    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        intensity = 255
        distance = 0
        
        while intensity > 0 and distance < max_length:
            nx = int(x + dx * distance)
            ny = int(y + dy * distance)
            
            if 0 <= nx < width and 0 <= ny < height:
                if channels == 1:  # Grayscale
                    if output_image[ny, nx] == 0:
                        break
                    # Combine ray intensity with random noise
                    output_image[ny, nx] = int(min(255, output_image[ny, nx] + intensity * (0.5 + 0.5 * random.random())))
                else:  # Color
                    if np.all(output_image[ny, nx] == [0, 0, 0]):
                        break
                    for c in range(channels):
                        output_image[ny, nx, c] = int(min(255, output_image[ny, nx, c] + intensity * (0.5 + 0.5 * random.random())))
            else:
                break
                
            distance += 1
            intensity *= intensity_decrease  # Decreasing intensity
    
    # Blend the original image with the image containing rays
    if channels == 1:
        output_image = np.round((1 - ratio) * image + ratio * output_image).astype(np.uint8)
    else:
        for c in range(channels):
            output_image[..., c] = np.round((1 - ratio) * image[..., c] + ratio * output_image[..., c]).astype(np.uint8)

    return output_image

def add_motion_artifact_CT(image, ratio):
    """
    Adds motion artifacts resembling parallel lines through non-black regions of the CT image.
    
    Parameters
    ----------
    image : np.ndarray
        The original CT image.
    ratio : float 
        A blending ratio between the original image and the image with artifacts. Also affects number of rays.
    
    Returns
    ----------
    output_image : np.ndarray 
        The image with added motion artifacts.
    """
    num_lines = 30
    intensity_decrease = 0.95

    output_image = image.copy()
    
    if len(image.shape) == 2:  # Grayscale image
        height, width = output_image.shape
        channels = 1
    else:  # Color image
        height, width, channels = output_image.shape
    
    # Generate number of lines drawn
    num_lines = round(ratio * 30)
    
    # Generate a random angle for the lines
    angle = random.uniform(0, np.pi)
    
    # Find all non-black pixels (assumed to be non-zero)
    if channels == 1:
        non_black_spots = np.argwhere(output_image != 0)
    else:
        non_black_spots = np.argwhere(np.any(output_image != [0, 0, 0], axis=-1))
    
    if non_black_spots.size == 0:
        return output_image
    
    for _ in range(num_lines):
        # Randomly choose a non-black spot to start the line
        chosen_spot = non_black_spots[random.randint(0, len(non_black_spots) - 1)]
        x, y = chosen_spot[1], chosen_spot[0]
        
        # Initialize intensity and distance for positive angle
        intensity = 255
        distance = 0
        
        # Positive angle direction
        dx_pos = np.cos(angle)
        dy_pos = np.sin(angle)
        
        while intensity > 0:
            nx_pos = int(x + distance * dx_pos)
            ny_pos = int(y + distance * dy_pos)
            
            if 0 <= nx_pos < width and 0 <= ny_pos < height:
                if channels == 1:  # Grayscale
                    if output_image[ny_pos, nx_pos] != 0:
                        output_image[ny_pos, nx_pos] = int(min(255, output_image[ny_pos, nx_pos] + intensity))
                    else:
                        break
                else:  # Color
                    if np.any(output_image[ny_pos, nx_pos] != [0, 0, 0]):
                        for c in range(channels):
                            output_image[ny_pos, nx_pos, c] = int(min(255, output_image[ny_pos, nx_pos, c] + intensity))
                    else:
                        break
            else:
                break
            
            distance += 1
            intensity *= intensity_decrease
        
        # Initialize intensity and distance for negative angle
        intensity = 255
        distance = 0
        
        # Negative angle direction (180 degrees opposite)
        dx_neg = -dx_pos
        dy_neg = -dy_pos
        
        while intensity > 0:
            nx_neg = int(x + distance * dx_neg)
            ny_neg = int(y + distance * dy_neg)
            
            if 0 <= nx_neg < width and 0 <= ny_neg < height:
                if channels == 1:  # Grayscale
                    if output_image[ny_neg, nx_neg] != 0:
                        output_image[ny_neg, nx_neg] = int(min(255, output_image[ny_neg, nx_neg] + intensity))
                    else:
                        break
                else:  # Color
                    if np.any(output_image[ny_neg, nx_neg] != [0, 0, 0]):
                        for c in range(channels):
                            output_image[ny_neg, nx_neg, c] = int(min(255, output_image[ny_neg, nx_neg, c] + intensity))
                    else:
                        break
            else:
                break
            
            distance += 1
            intensity *= intensity_decrease
    
    # Blend the original image with the image containing artifacts
    if channels == 1:
        output_image = np.round((1 - ratio) * image + ratio * output_image).astype(np.uint8)
    else:
        for c in range(channels):
            output_image[..., c] = np.round((1 - ratio) * image[..., c] + ratio * output_image[..., c]).astype(np.uint8)

    return output_image

def add_motion_artifact_MRI(image, ratio):
    """
    Adds motion artifacts to an MRI image by shifting non-black pixels uniformly in both horizontal and vertical directions.
    
    Parameters
    ----------
    image : np.ndarray 
        The original MRI image.
    ratio : float 
        A blending ratio between the original image and the image with artifacts.
    
    Returns
    ----------
    output_image : np.ndarray 
        The image with added motion artifacts.
    """
    shift_amount = 100

    output_image = image.copy().astype(np.float32)  # Convert to float for blending
    height, width = output_image.shape[:2]  # Get height and width
    
    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1  # Grayscale image
    
    # Create copies for artifacts overlay
    artifacts_image_1 = np.zeros_like(output_image).astype(np.float32)
    artifacts_image_2 = np.zeros_like(output_image).astype(np.float32)
    
    # Find all non-black pixels (assumed to be non-zero)
    if channels == 1:
        non_black_spots = np.argwhere(output_image != 0)
    else:
        non_black_spots = np.argwhere(np.any(output_image != [0, 0, 0], axis=-1))
    
    if non_black_spots.size == 0:
        return output_image.astype(np.uint8)
    
    # Randomly choose shift amounts for horizontal and vertical directions
    shift_x = random.randint(-shift_amount, shift_amount)
    shift_y = random.randint(-shift_amount, shift_amount)
    
    for y, x in non_black_spots:
        # Calculate new positions with uniform shift
        new_x_1 = x + shift_x
        new_y_1 = y + shift_y
        new_x_2 = x - shift_x
        new_y_2 = y - shift_y
        
        # Ensure new coordinates are within bounds for artifacts_image_1
        if 0 <= new_x_1 < width and 0 <= new_y_1 < height:
            if channels == 1:  # Grayscale
                artifacts_image_1[new_y_1, new_x_1] = image[y, x]
            else:  # Color
                artifacts_image_1[new_y_1, new_x_1, :] = image[y, x, :]
        
        # Ensure new coordinates are within bounds for artifacts_image_2
        if 0 <= new_x_2 < width and 0 <= new_y_2 < height:
            if channels == 1:  # Grayscale
                artifacts_image_2[new_y_2, new_x_2] = image[y, x]
            else:  # Color
                artifacts_image_2[new_y_2, new_x_2, :] = image[y, x, :]
    
    # Blend artifacts images with original image
    artifacts_blend = ((artifacts_image_1 * 0.33) + (artifacts_image_2 * 0.33) + (image * 0.34)).astype(np.uint8)
    
    # Blend the original image with the image containing artifacts
    output_image = np.round((1 - ratio) * image + ratio * artifacts_blend).astype(np.uint8)

    return output_image

def add_motion_artifact_PET(image, ratio):
    """
    Adds motion artifacts to an MRI image by shifting non-black pixels uniformly in both horizontal and vertical directions.
    
    Parameters
    ----------
    image : np.ndarray 
        The original PET image.
    ratio : float 
        A blending ratio between the original image and the image with artifacts.
    
    Returns
    ----------
    output_image : np.ndarray 
        The image with added motion artifacts.
    """
    shift_amount = 3

    output_image = image.copy().astype(np.float32)  # Convert to float for blending
    height, width = output_image.shape[:2]  # Get height and width
    
    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1  # Grayscale image
    
    # Create copies for artifacts overlay
    artifacts_image_1 = np.zeros_like(output_image).astype(np.float32)
    artifacts_image_2 = np.zeros_like(output_image).astype(np.float32)
    
    # Find all non-black pixels (assumed to be non-zero)
    if channels == 1:
        non_black_spots = np.argwhere(output_image != 0)
    else:
        non_black_spots = np.argwhere(np.any(output_image != [0, 0, 0], axis=-1))
    
    if non_black_spots.size == 0:
        return output_image.astype(np.uint8)
    
    # Randomly choose shift amounts for horizontal and vertical directions
    shift_x = random.randint(-shift_amount, shift_amount)
    shift_y = random.randint(-shift_amount, shift_amount)
    
    for y, x in non_black_spots:
        # Calculate new positions with uniform shift
        new_x_1 = x + shift_x
        new_y_1 = y + shift_y
        new_x_2 = x - shift_x
        new_y_2 = y - shift_y
        
        # Ensure new coordinates are within bounds for artifacts_image_1
        if 0 <= new_x_1 < width and 0 <= new_y_1 < height:
            if channels == 1:  # Grayscale
                artifacts_image_1[new_y_1, new_x_1] = image[y, x]
            else:  # Color
                artifacts_image_1[new_y_1, new_x_1, :] = image[y, x, :]
        
        # Ensure new coordinates are within bounds for artifacts_image_2
        if 0 <= new_x_2 < width and 0 <= new_y_2 < height:
            if channels == 1:  # Grayscale
                artifacts_image_2[new_y_2, new_x_2] = image[y, x]
            else:  # Color
                artifacts_image_2[new_y_2, new_x_2, :] = image[y, x, :]
    
    # Blend artifacts images with original image
    artifacts_blend = ((artifacts_image_1 * 0.33) + (artifacts_image_2 * 0.33) + (image * 0.34)).astype(np.uint8)
    
    # Blend the original image with the image containing artifacts
    output_image = np.round((1 - ratio) * image + ratio * artifacts_blend).astype(np.uint8)

    return output_image

def add_gibbs_artifact_MRI(image, ratio):
    """
    Adds Gibbs artifacts to an MRI image.
    
    Parameters
    ----------
    image : np.ndarray 
        The original MRI image.
    ratio : float 
        A blending ratio between the original image and the image with artifacts.
    
    Returns
    ----------
    output_image : np.ndarray 
        The image with added Gibbs artifacts.
    """
    amplitude = 200

    output_image = image.copy().astype(np.float32)  # Convert to float for blending
    
    if len(image.shape) == 3:  # Color image
        height, width, channels = image.shape
        gradient_magnitude = np.zeros((height, width, channels), dtype=np.float32)
        
        for c in range(channels):
            # Calculate gradient magnitude (edge strength) for each channel
            gradient = np.gradient(output_image[..., c])
            gradient_magnitude[..., c] = np.sqrt(gradient[0]**2 + gradient[1]**2)
        
        # Normalize gradient magnitude to [0, 1] for each channel
        gradient_magnitude /= gradient_magnitude.max(axis=(0, 1))
        
        # Create Gibbs oscillations proportional to the gradient magnitude for each channel
        gibbs_artifact = amplitude * gradient_magnitude
        
        output_image += gibbs_artifact
        
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        
        # Blend the original image with the image containing artifacts
        for c in range(channels):
            output_image[..., c] = np.round((1 - ratio) * image[..., c] + ratio * output_image[..., c]).astype(np.uint8)
    
    else:  # Grayscale image
        height, width = image.shape
        
        # Calculate gradient magnitude (edge strength)
        gradient = np.gradient(output_image)
        gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
        
        # Normalize gradient magnitude to [0, 1]
        gradient_magnitude /= gradient_magnitude.max()
        
        # Create Gibbs oscillations proportional to the gradient magnitude
        gibbs_artifact = amplitude * gradient_magnitude
        
        output_image += gibbs_artifact
        
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        
        # Blend the original image with the image containing artifacts
        output_image = np.round((1 - ratio) * image + ratio * output_image).astype(np.uint8)

    return output_image

def add_chemical_shift_artifact_MRI(image, ratio):
    """
    Adds a chemical shift artifact to an MRI image by dilating the black edges and selectively blurring them.

    Parameters
    ----------
    image : np.ndarray 
        The original MRI image.
    ratio : float 
        A blending ratio between the original image and the image with artifacts.

    Returns
    ----------
    output_image : np.ndarray 
        The MRI image with added chemical shift artifact.
    """
    dilation_pixels = 3
    
    if len(image.shape) == 2:  # Grayscale image
        channels = 1
    else:  # Color image
        channels = image.shape[2]

    if channels == 1:  # Grayscale image
        # Threshold the image to get a binary mask of the background (black regions)
        _, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate the binary mask to expand the black regions
        kernel = np.ones((dilation_pixels, dilation_pixels), np.uint8)
        dilated_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
        
        # Apply the dilated mask to the original image
        artifact_image = np.where(dilated_mask == 255, 0, image)
        
        # Gaussian blur only on the dilated areas
        dilated_indices = np.where(dilated_mask == 255)
        blurred_image = cv2.GaussianBlur(artifact_image.astype(np.float32), (9, 9), 0)
        artifact_image[dilated_indices] = blurred_image[dilated_indices]
        
        # Blend the original image with the image containing artifacts
        output_image = np.round((1 - ratio) * image + ratio * artifact_image).astype(np.uint8)
    
    else:  # Color image
        # Create a binary mask for each channel and combine them
        binary_masks = []
        for c in range(channels):
            _, binary_mask = cv2.threshold(image[..., c], 1, 255, cv2.THRESH_BINARY_INV)
            binary_masks.append(binary_mask)
        combined_binary_mask = np.min(binary_masks, axis=0)

        # Dilate the combined binary mask to expand the black regions
        kernel = np.ones((dilation_pixels, dilation_pixels), np.uint8)
        dilated_mask = cv2.dilate(combined_binary_mask.astype(np.uint8), kernel, iterations=1)
        
        # Apply the dilated mask to the original image
        artifact_image = np.where(dilated_mask[..., None] == 255, 0, image)
        
        # Gaussian blur only on the dilated areas
        dilated_indices = np.where(dilated_mask == 255)
        blurred_image = cv2.GaussianBlur(artifact_image.astype(np.float32), (9, 9), 0)
        for c in range(channels):
            artifact_image[dilated_indices + (c,)] = blurred_image[dilated_indices + (c,)]
        
        # Blend the original image with the image containing artifacts
        output_image = np.zeros_like(image, dtype=np.uint8)
        for c in range(channels):
            output_image[..., c] = np.round((1 - ratio) * image[..., c] + ratio * artifact_image[..., c]).astype(np.uint8)

    return output_image

def add_attenuation_artifact_PET(image, ratio):
    """
    Adds a white spot with a soft transition at a random position in the image.

    Parameters
    ----------
    image : np.ndarray 
        The original PET image. Can be grayscale or color.
    ratio : float 
        The ratio used to determine the size of the white spot relative to the image dimensions.
    
    Returns:
        output_image : np.ndarray 
            The image with the inserted white spot.
    """
    output_image = image.copy()
    if len(image.shape) == 2:  # Grayscale image
        height, width = output_image.shape
        channels = 1
    else:  # Color image
        height, width, channels = output_image.shape
    
    # Randomly select a position for the white spot that is not black
    while True:
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        if len(image.shape) == 2:  # Grayscale image
            if output_image[center_y, center_x] != 0:  # Check if the pixel is not black
                break
        else:  # Color image
            if np.any(output_image[center_y, center_x] != 0):  # Check if the pixel is not black
                break
    
    # Determine the radius of the white spot based on the provided ratio
    radius = int(3 * ratio)
    
    # Create a circular mask with a soft transition
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.circle(mask, (center_x, center_y), radius, (1.0,), thickness=-1)
    mask = cv2.GaussianBlur(mask, (5, 5), 10)  # Soft edges using Gaussian Blur
    
    # Normalize the mask to the range 0-1
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    
    if len(image.shape) == 2:  # Grayscale image
        # Apply the mask to the image
        output_image = (output_image + (255 - output_image) * mask).astype(np.uint8)
    else:  # Color image
        for c in range(channels):
            output_image[..., c] = (output_image[..., c] + (255 - output_image[..., c]) * mask).astype(np.uint8)
    
    return output_image

def add_beam_hardening_artifact_CT(image, ratio):
    """
    Adds subtle dark bands to simulate beam-hardening artifacts in a CT image.
    
    Parameters 
    ----------
        image : np.ndarray 
            The original CT image. Can be grayscale or color.
        ratio : float 
            The ratio used to determine the intensity of the bands.
    
    Returns 
    ----------
        output_image : np.ndarray 
            The image with added beam-hardening artifacts.
    """
    num_bands = 10
    band_width = 5
    band_intensity = 0.5 * ratio

    output_image = image.copy().astype(np.float32)
    
    if len(output_image.shape) == 2:  # Grayscale image
        height, width = output_image.shape
        channels = 1
    else:  # Color image
        height, width, channels = output_image.shape

    for _ in range(num_bands):
        # Randomly select positions and angle for the dark band
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        angle = np.random.uniform(0, np.pi)
        length = max(width, height)
        
        # Calculate the end point of the band based on the angle
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # Create a mask for the dark band
        band_mask = np.zeros((height, width), dtype=np.float32)
        cv2.line(band_mask, (x1, y1), (x2, y2), (1,), thickness=band_width)
        band_mask = cv2.GaussianBlur(band_mask, (21, 21), 10)  # Soft edges using Gaussian Blur
        
        # Apply the band mask to darken the image
        if channels == 1:  # Grayscale image
            output_image -= band_mask * (1 - band_intensity) * 255
        else:  # Color image
            for c in range(channels):
                output_image[..., c] -= band_mask * (1 - band_intensity) * 255

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image

def add_ring_artifact_CT(image, ratio):
    """
    Adds a ring artifact to a CT image.
    
    Parameters 
    ----------
    image : np.ndarray 
        The original CT image. Can be grayscale or color.
    ratio : float
        The ratio used to determine the intensity of the ring.        
    
    Returns 
    ----------
    output_image : np.ndarray 
        The image with an added ring artifact.
    """
    num_rings = 1
    ring_intensity = 100 * ratio

    output_image = image.copy().astype(np.float32)
    
    if len(image.shape) == 2:  # Grayscale image
        height, width = output_image.shape
        channels = 1
    else:  # Color image
        height, width, channels = output_image.shape
    
    center_x, center_y = width // 2, height // 2
    max_radius = min(center_x, center_y) // 4

    for _ in range(num_rings):
        # Randomly select radius for the ring
        radius = np.random.randint(max_radius // 4, max_radius)
        thickness = np.random.randint(1, 3)

        # Create a circular ring mask
        ring_mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(ring_mask, (center_x, center_y), radius, (1,), thickness=thickness)
        ring_mask = cv2.GaussianBlur(ring_mask, (5, 5), 1)  # Soft edges using Gaussian Blur
        
        # Apply the ring mask to darken or brighten the image
        if channels == 1:  # Grayscale image
            output_image += ring_mask * ring_intensity * (np.random.rand() * 0.5 + 0.5)
        else:  # Color image
            for c in range(channels):
                output_image[..., c] += ring_mask * ring_intensity * (np.random.rand() * 0.5 + 0.5)

    # Clip values to valid range and convert back to uint8
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image

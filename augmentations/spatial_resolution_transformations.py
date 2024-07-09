import numpy as np
import cv2
from ..enums import Input_Modality, Output_Modality


def transform_spatial_resolution(image_to_be_transformed: np.ndarray, ratio: float, input_modality: Input_Modality, output_modality: Output_Modality):
    '''
    Adjusts spatial resolution of an image by blurring or sharpening it.

    Knowledge about spatial resolutions is taken from:
    - Kasban, Hany, M. A. M. El-Bendary, and D. H. Salama. "A comparative study of medical imaging techniques." International Journal of Information Science and Intelligent System 4.2 (2015): 37-58. 
    - Key, Jaehong, and James F. Leary. "Nanoparticles for multimodal in vivo imaging in nanomedicine." International journal of nanomedicine (2014): 711-726. 
    - Yim, Hyeona, Seogjin Seo, and Kun Na. "MRI Contrast Agent‐Based Multifunctional Materials: Diagnosis and Therapy." Journal of Nanomaterials 2011.1 (2011): 747196. 

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.
    input_modality : Input_Modality
        The modality of the image_to_be_transformed. If Input_Modality.any is 
        chosen, there are no modality-specific adjustments.
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
                    #pass
                    return blur_image(image_to_be_transformed, ratio)
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
                    return sharpen_image(image_to_be_transformed, ratio)
                case Output_Modality.CT:
                    return sharpen_image(image_to_be_transformed, ratio)
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case Input_Modality.MRI:
            match output_modality:
                case Output_Modality.PET:
                    return blur_image(image_to_be_transformed, ratio)
                case Output_Modality.MRI:
                    pass
                case Output_Modality.CT:
                    return blur_image(image_to_be_transformed, ratio*0.4)
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case Input_Modality.CT:
            match output_modality:
                case Output_Modality.PET:
                    return blur_image(image_to_be_transformed, ratio)
                case Output_Modality.MRI:
                    return sharpen_image(image_to_be_transformed, ratio*0.4)
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



def blur_image(image_to_be_transformed : np.ndarray, ratio : float): 
    """
    Blurrs an image by applying a Gaussian Blur.

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.

    Returns
    -------
    blurred_image : np.ndarray
        Transformed input image.
    """
    kernel_size = (0, 0)
    if ratio < 0.17:
        kernel_size = (3, 3)
    elif ratio < 34:
        kernel_size = (5, 5)
    elif ratio < 51:
        kernel_size = (7, 7)
    elif ratio < 67:
        kernel_size = (9, 9)
    elif ratio < 85:
        kernel_size = (11, 11)
    else:
        kernel_size = (13, 13)
    
    blurred_image = cv2.GaussianBlur(image_to_be_transformed, kernel_size, 0)

    return blurred_image

def sharpen_image(image_to_be_transformed : np.ndarray, ratio : float):
    """
    Sharpens an image by applying a 2D filter.

    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        Ratio influencing the sharpening kernel. Must be between 0 and 1.

    Returns
    -------
    sharpened_image : np.ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the given ratio is <0 or >1.
    """
    if ratio < 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1.")
    
    kernel_base = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel_addition = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel_final = kernel_base + (ratio * kernel_addition)

    sharpened_image = cv2.filter2D(image_to_be_transformed, -1, kernel_final)

    return sharpened_image

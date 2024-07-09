import numpy as np
from ..enums import Input_Modality, Output_Modality


def transform_noise(image_to_be_transformed: np.ndarray, ratio: float, input_modality: Input_Modality, output_modality: Output_Modality):
    '''
    Adds modality characteristic noise to a given image.

    Knowledge about modality characteristic noise is taken from:
    - Aja-Fernández, Santiago, and Gonzalo Vegas-Sánchez-Ferrero. "Statistical analysis of noise in MRI." Switzerland: Springer International Publishing (2016).
    - Diwakar, Manoj, and Manoj Kumar. "A review on CT image noise and its denoising." Biomedical Signal Processing and Control 42 (2018): 73-88. 
    - Wang, Jing, et al. "An experimental study on the noise properties of x-ray CT sinogram data in Radon space." Physics in Medicine & Biology 53.12 (2008): 3327.
    - Erlangen-Nürnberg, Lehrstuhl Für Mustererkennung Friedrich-Alexander-Universität. “CT Image De-Noising.” Copyright (C) LME, www5.cs.fau.de/en/our-team/balda-michael/projects/ct-image-de-noising/index.html#:~:text=The%20two%20main%20sources%20of,is%20square%2Droot%20of%20m. 
    - Kim, Ji Hye, et al. "Post-filtering of PET image based on noise characteristic and spatial sensitivity distribution." 2013 IEEE Nuclear Science Symposium and Medical Imaging Conference (2013 NSS/MIC). IEEE, 2013. 

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
                    return add_gaussian_noise(image_to_be_transformed, ratio)
                case Output_Modality.MRI:
                    return add_rician_noise(image_to_be_transformed, ratio)
                case Output_Modality.CT:
                    return add_poisson_noise(image_to_be_transformed, ratio)
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case Input_Modality.PET:
            match output_modality:
                case Output_Modality.PET:
                    pass
                case Output_Modality.MRI:
                    return add_rician_noise(image_to_be_transformed, ratio)
                case Output_Modality.CT:
                    return add_poisson_noise(image_to_be_transformed, ratio)
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case Input_Modality.MRI:
            match output_modality:
                case Output_Modality.PET:
                    return add_gaussian_noise(image_to_be_transformed, ratio)
                case Output_Modality.MRI:
                    pass
                case Output_Modality.CT:
                    return add_poisson_noise(image_to_be_transformed, ratio)
                case Output_Modality.custom:
                    pass
                case _:
                    raise ValueError("Output modality is not implemented.")
        case Input_Modality.CT:
            match output_modality:
                case Output_Modality.PET:
                    return add_gaussian_noise(image_to_be_transformed, ratio)
                case Output_Modality.MRI:
                    return add_rician_noise(image_to_be_transformed, ratio)
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



def add_gaussian_noise(image_to_be_transformed : np.ndarray, ratio : float):
    """
    Adds Gaussian noise to a given image to simulate PET noise.
    
    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.

    Returns
    -------
    noisy_image : np.ndarray
        Transformed input image.
    """
    # Calculate noise strength
    noise_strength = 10*ratio

    # Create Gaussian Noise
    noise = np.random.normal(0, noise_strength, image_to_be_transformed.shape)

    # Clip the values to ensure they are within the valid range
    noisy_image = np.clip(image_to_be_transformed + noise, 0, 255).astype(np.uint8)

    return noisy_image

def add_rician_noise(image_to_be_transformed : np.ndarray, ratio : float):
    """
    Adds Rician noise to a given image to simulate MRI noise.
    
    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.

    Returns
    -------
    noisy_image : np.ndarray
        Transformed input image.
    """
    # Calculate noise strength
    noise_strength = int(5 * ratio)

    # Generate Gaussian noise
    noise_real = np.random.normal(0, noise_strength, image_to_be_transformed.shape)
    noise_imag = np.random.normal(0, noise_strength, image_to_be_transformed.shape)
    
    # Rician noise is the magnitude of the real and imaginary Gaussian noise
    rician_noise = np.sqrt((image_to_be_transformed + noise_real) ** 2 + noise_imag ** 2)
    
    # Normalize the noisy image to keep the values between 0 and 255
    noisy_image = np.clip(rician_noise, 0, 255).astype(np.uint8)
    
    return noisy_image

def add_poisson_noise(image_to_be_transformed : np.ndarray, ratio : float):
    """
    Adds Poisson noise to a given image to simulate CT noise.
    
    Parameters
    ----------
    image_to_be_transformed : np.ndarray
        Input image. Can be gray-scale or in color.
    ratio : float
        The ratio of adjustment. A value of 0 means no adjustment, and 1 means
        fully adjusted.

    Returns
    -------
    noisy_image : np.ndarray
        Transformed input image.
    """
    image = image_to_be_transformed.astype(np.float32) 

    # Scale the image to the range [0, 1] 
    image_min, image_max = image.min(), image.max() 
    scaled_image = (image - image_min) / (image_max - image_min) 

    # Generate Poisson noise 
    poisson_noise = np.random.poisson(0.5* scaled_image * 255.0) / (0.5* 255.0) 

    # Blend the original image with the noise based on the ratio 
    noisy_image = (1 - ratio) * scaled_image + ratio * poisson_noise 

    # Scale back to the original range 
    noisy_image = noisy_image * (image_max - image_min) + image_min 

    # Clip the values to the valid range for image data types 
    noisy_image = np.clip(noisy_image, image_min, image_max) 

    # Convert back to the original data type 
    noisy_image = noisy_image.astype(np.uint8) 

    return noisy_image

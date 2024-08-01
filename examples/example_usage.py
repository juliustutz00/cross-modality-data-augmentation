from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.utils.image_utils import load_npy_image, save_npy_image, display_image_gray
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality



# Load input image
image_to_be_transformed = load_npy_image("folder/folder/filename.npy")

# Create transformation instance
cross_modality_transformer = CrossModalityTransformations(
    input_modality=Input_Modality.CT, 
    output_modality=Output_Modality.MRI, 
    transformation_probability=0.2,     # probability that an image is transformed at all
    atLeast = 2,    # minimal number of augmentations
    atMost = 4,    # maximal number of augmentations
    color_probability = 1.0,     # probability for the color augmentation to be chosen from the sampled number of overall augmentations
    artifact_probability = 0.05, 
    spatial_resolution_probability = 0.8, 
    noise_probability = 0.2,
    color_ratio_range=(0, 1),     # strength of the color augmentation (if it is chosen)
    artifact_ratio_range=(0, 1), 
    spatial_resolution_ratio_range=(0, 1), 
    noise_ratio_range=(0, 1)
)

# Apply transformations
output_image = cross_modality_transformer.transform(image_to_be_transformed)

# Display and save transformed image
display_image_gray(output_image)
save_npy_image("folder/folder/filename.npy", output_image)

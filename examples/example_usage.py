from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.utils.image_utils import load_npy_image, save_npy_image, display_image_gray
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality



# Load input image
image_to_be_transformed = load_npy_image("folder/folder/filename.npy")

# Create transformation instance
cross_modality_transformer = CrossModalityTransformations(
    input_modality=Input_Modality.CT, 
    output_modality=Output_Modality.MRI, 
    color_ratio_range=(0, 1), 
    artifact_ratio_range=(0, 1), 
    spatial_resolution_ratio_range=(0, 1), 
    noise_ratio_range=(0, 1)
)

# Apply transformations
output_image = cross_modality_transformer.transform(image_to_be_transformed)

# Display and save transformed image
display_image_gray(output_image)
save_npy_image("folder/folder/filename.npy", output_image)

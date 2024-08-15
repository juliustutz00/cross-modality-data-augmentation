from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.utils.image_utils import load_npy_image, load_custom_dataset
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality



# Create a reference image of a custom modality by using a given dataset of that custom modality
# The newly created reference image will be located at the folder "cross_modality_data_augmentation/reference_images/custom"
load_custom_dataset("path/to/your/dataset", "custom_modality_name") 

# Load input image
image_to_be_transformed = load_npy_image("folder/folder/filename.npy")

# Create transformation instance
cross_modality_transformer = CrossModalityTransformations(
    input_modality=Input_Modality.any, 
    output_modality=Output_Modality.custom, 
    custom_reference_image_name="custom_modality_name_256x256.npy"
)

# Apply transformations
output_image = cross_modality_transformer.transform(image_to_be_transformed)

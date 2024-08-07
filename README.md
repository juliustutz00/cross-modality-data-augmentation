
# Cross-Modality Data Augmentation

This robust, cross-modality data augmentation technique is capable of synthesizing new medical images from a given to a desired domain. 

The thereby newly created training samples can help to improve the generalization performance of deep learning models when training with multiple modalities. 




## Implemented Modalities

The Data Augmentation is fine-tuned for the following modalities:
- PET
- MRI
- CT

It is, however, possible to add custom modalities, although their transformation will be rather coarse. More information is given in __Custom Modalities__.
## Custom Modalities

It is possible to implement custom modalities yourself, altough the corresponding transformation will be rather coarse because of the missing fine-tuning.

To add a new modality proceed as follows:
- Pre-process dataset: The augmentation needs a dataset of the destination modality to work (e.g. if you want to transform MRI to US, you need a sufficiently large (>= 200 images) US dataset). This dataset should be pre-processed, meaning: .npy format, same quadratic shape, removed noise, similar size dimensions.
- Import function: Import the cross_modality_data_augmentation.utils.image_utils.load_custom_dataset function.
- Create custom reference image: Use the imported load_custom_dataset function where folder_path: str should be the path of the folder where your dataset is located and modality:str should be the name of the dataset's modality.
- Use new modality: You can now use your custom modality by creating a new transformer (more information is given in __Usage__) with output_modality=Output_Modality.custom and custom_reference_image_name=(modality + "_256x256.npy").
## Usage

cross_modality_transformer = CrossModalityTransformations(
    input_modality=Input_Modality.CT,   # set the initial modality of the image you want to augment
    output_modality=Output_Modality.MRI,   # set the destination modality of the image you want to augment
    transformation_probability=0.2,     # probability that an image is augmented at all
    atLeast = 2,    # minimal number of augmentations
    atMost = 4,    # maximal number of augmentations
    color_probability = 1.0,     # probability for the color augmentation to be chosen from the sampled number of overall augmentations
    artifact_probability = 0.05, 
    spatial_resolution_probability = 0.8, 
    noise_probability = 0.2,
    color_ratio_range=(0, 1),     # strength of the color augmentation (if it is chosen)
    artifact_ratio_range=(0, 1), 
    spatial_resolution_ratio_range=(0, 1), 
    noise_ratio_range=(0, 1), 
    custom_reference_image_name=None   # name of the custom image if custom is chosen as output_modality
)

The usage is kept similar to existing data augmentations. One can set a probability for the whole data augmentation to happen, and probabilities + ratios for every single augmentation. The cross-modality data augmentation can also easily be included in pipelines of other data augmentations. For further information on the usage please have a look at the folder __examples__.
## Permitted values

input_modality: cross_modality_data_augmentation.enums.Input_Modality
output_modality: cross_modality_data_augmentation.enums.Output_Modality
transformation_probability: [0, 1]
atLeast = [0, 4]
atMost = [atLeast, 4]
color_probability = [0, 1]
artifact_probability = [0, 1]
spatial_resolution_probability = [0, 1]
noise_probability = [0, 1]
color_ratio_range= [0, 1]
artifact_ratio_range= [0, 1]
spatial_resolution_ratio_range= [0, 1]
noise_ratio_range= [0, 1]
custom_reference_image_name: str

The input images should be pre-processed, meaning: .npy format, same shape, removed noise, similar size dimensions.
If a non-implemented modality is augmented, choose input_modality=Input_Modality.any. 
If a non-implemented modality is chosen as a destination, choose output_modality=Output_Modality.custom and custom_reference_image_name="your_reference_image.png" (more information is given in __Custom Modalities__).

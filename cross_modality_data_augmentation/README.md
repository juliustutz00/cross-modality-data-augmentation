
# Medical Data Augmentation

This robust, cross-modality-wise Data Augmentation technique is capable of synthesizing new medical images from a given to a desired domain. 

The thereby newly created training samples can help to improve the generalization performance of deep learning models when training with multiple modalities. 




## Implemented Modalities

The Data Augmentation is fine-tuned for the following modalities:
- PET
- MRI
- CT
It is, however, possible to add custom modalities, although their transformation will be rather coarse. More information is given in __Custom Modalities__.
## GUI and permitted values

To improve the usability of this data augmentation it is possible to augment datasets via a GUI. The augmentation only works with certain prerequisites which, as well as the explanation of the GUI can be seen in the following:
- "Dataset input": Choose the folder with your dataset inside. Permitted values: valid folder name with dataset; dataset has to be prepared, meaning dicom (.dcm) format, same quadratic shape, removed noise, similar size ratio
- "From": Choose the modality of your put in dataset. If none of the dropdown menu's modalities is yours, choose "any". Permitted values: any value from the dropdown menu
- "To": Choose the modality that you want to transform your put in dataset to. If none of the dropdown menu's modalities is the one you aim for you can upload a dataset of your aimed for modality. For this see __Custom Modalities__. Permitted values: any value from the dropdown menu
- "Custom": Choose your custom modality that you implemeted yourself as described in __Custom Modalities__. This row is only accessible if "custom" is chosen in row "_To_". Permitted values: any file from the dropdown menu; file must have the same resolution as the put in dataset in "_Dataset input_"
- "Approximate augmentations per image": Choose the approximate augmentations per image in the dataset. Permitted values: any integer >0; higher values will be computationally expensive
- "Add custom modality (optional)": Implement your own custom modality as described in __Custom Modalities__
- "Augment": This button will start the augmentation with the given values. The status label informs you about the current status
## Custom Modalities

It is possible to implement custom modalities yourself, altough the corresponding transformation will be rather rough because of the missing fine-tuning.

To add a new modality proceed as follows:
- Prepare dataset: The transformation needs a dataset of the desired domain to work (e.g. if you want to transform MRI to US, you need a sufficiently large US dataset). This dataset should be prepared, meaning: dicom (.dcm) format, same quadratic shape, removed noise, similar size ratio.
- Open GUI: Open the GUI as usual.
- Upload dataset: Concentrate on the row "Add custom modality (optional)". First, choose a suitable resoluion. This resolution must _not_ be the same as the dataset but it should be the same resolution as the dataset you want to augment (e.g. if you want to transform MRI to US, you need to choose the resolution of the MRI dataset). Click on "Choose Folder" and select the folder of the dataset of the custom modality.
- Use new modality: You can now use your custom modality, _without_ having to restart the GUI. Therefore select "custom" in row "To" which should spawn a new row "Custom". In row "Custom" please select your custom modality from the dropdown menu. It will have the same name as the folder you inserted at step "_Upload dataset_" followed by an underscore, the chosen resolution and .npy (e.g. if your dataset is called "US_images" and you chose the resolution 256x256, the custom modality file will be called "US_images_256x256.npy").
import numpy as np
import random
import os
import pkg_resources
from .enums import Input_Modality, Output_Modality
from .augmentations import (
    color_transformations,
    artifact_transformations, 
    spatial_resolution_transformations,
    noise_transformations
)


class CrossModalityTransformations:
    def __init__(self, input_modality: Input_Modality, output_modality: Output_Modality, color_ratio_range=(0, 1), artifact_ratio_range=(0, 1), spatial_resolution_ratio_range=(0, 1), noise_ratio_range=(0, 1), custom_reference_image_name=None):
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.color_ratio_range = color_ratio_range
        self.artifact_ratio_range = artifact_ratio_range
        self.spatial_resolution_ratio_range = spatial_resolution_ratio_range
        self.noise_ratio_range = noise_ratio_range
        self.custom_reference_image_name = custom_reference_image_name

    def random_ratio(self, ratio_range):
        return random.uniform(ratio_range[0], ratio_range[1])

    def determine_reference_image(self, image: np.ndarray, output_modality: Output_Modality, custom_reference_image_name):
        if (not (output_modality is Output_Modality.custom)) or custom_reference_image_name is None:
            shape_number_last = image.shape[-1]
            shape_number_second_to_last = image.shape[-2]
            files = os.listdir(pkg_resources.resource_filename(__name__, "reference_images/built_in"))
            matching_files = [f for f in files if f.endswith(f"{shape_number_second_to_last}x{shape_number_last}.npy") and f.startswith(output_modality.name)]
            if len(matching_files) != 1:
                raise ValueError(f"Expected exactly one file starting with '{output_modality.name} and ending with {shape_number_second_to_last}x{shape_number_last}.npy', found {len(matching_files)}")
            return np.load(os.path.join(pkg_resources.resource_filename(__name__, "reference_images/built_in"), matching_files[0]))
        else:
            return np.load(os.path.join(pkg_resources.resource_filename(__name__, "reference_images/custom"), custom_reference_image_name))
            
    def transform(self, image: np.ndarray):
        color_ratio = self.random_ratio(self.color_ratio_range)
        artifact_ratio = self.random_ratio(self.artifact_ratio_range)
        spatial_resolution_ratio = self.random_ratio(self.spatial_resolution_ratio_range)
        noise_ratio = self.random_ratio(self.noise_ratio_range)
        reference_image = self.determine_reference_image(image, self.output_modality, self.custom_reference_image_name)

        image = color_transformations.transform_color(image, reference_image, color_ratio, self.input_modality, self.output_modality)
        image = artifact_transformations.transform_artifact(image, artifact_ratio, self.input_modality, self.output_modality)
        image = spatial_resolution_transformations.transform_spatial_resolution(image, spatial_resolution_ratio, self.input_modality, self.output_modality)
        image = noise_transformations.transform_noise(image, noise_ratio, self.input_modality, self.output_modality)

        return image

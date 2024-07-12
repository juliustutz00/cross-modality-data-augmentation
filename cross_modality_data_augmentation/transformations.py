import numpy as np
import random
import os
from .enums import Input_Modality, Output_Modality
from .augmentations import (
    color_transformations,
    artifact_transformations, 
    spatial_resolution_transformations,
    noise_transformations
)


class CrossModalityTransformations:
    def __init__(self, input_modality: Input_Modality, output_modality: Output_Modality, 
                 atLeast=0, 
                 atMost=4,
                 color_probability=1.0, color_ratio_range=(0, 1), 
                 artifact_probability=1.0, artifact_ratio_range=(0, 1), 
                 spatial_resolution_probability=1.0, spatial_resolution_ratio_range=(0, 1), 
                 noise_probability=1.0, noise_ratio_range=(0, 1), 
                 custom_reference_image_name=None):
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.atLeast = atLeast
        self.atMost = atMost
        self.color_probability = color_probability
        self.color_ratio_range = color_ratio_range
        self.artifact_probability = artifact_probability
        self.artifact_ratio_range = artifact_ratio_range
        self.spatial_resolution_probability = spatial_resolution_probability
        self.spatial_resolution_ratio_range = spatial_resolution_ratio_range
        self.noise_probability = noise_probability
        self.noise_ratio_range = noise_ratio_range
        self.custom_reference_image_name = custom_reference_image_name

    def random_ratio(self, ratio_range):
        return random.uniform(ratio_range[0], ratio_range[1])

    def determine_reference_image(self, image: np.ndarray, output_modality: Output_Modality, custom_reference_image_name):
        if (not (output_modality is Output_Modality.custom)) or custom_reference_image_name is None:
            shape_number_last = image.shape[-1]
            shape_number_second_to_last = image.shape[-2]
            files = os.listdir("reference_images/built_in")
            matching_files = [f for f in files if f.endswith(f"{shape_number_second_to_last}x{shape_number_last}.npy") and f.startswith(output_modality.name)]
            if len(matching_files) != 1:
                raise ValueError(f"Expected exactly one file starting with '{output_modality.name} and ending with {shape_number_second_to_last}x{shape_number_last}.npy', found {len(matching_files)}")
            return np.load(os.path.join("reference_images/built_in", matching_files[0]))
        else:
            return np.load(os.path.join("reference_images/custom", custom_reference_image_name))

    def transform(self, image: np.ndarray):
        color_ratio = self.random_ratio(self.color_ratio_range)
        artifact_ratio = self.random_ratio(self.artifact_ratio_range)
        spatial_resolution_ratio = self.random_ratio(self.spatial_resolution_ratio_range)
        noise_ratio = self.random_ratio(self.noise_ratio_range)
        reference_image = self.determine_reference_image(image, self.output_modality, self.custom_reference_image_name)

        if not (0 <= self.color_probability <= 1 and
            0 <= self.artifact_probability <= 1 and
            0 <= self.spatial_resolution_probability <= 1 and
            0 <= self.noise_probability <= 1):
            raise ValueError("Probability must be in range (0, 1).")
        
        if not (0 <= self.atLeast <= 4 and 0 <= self.atMost <= 4 and self.atLeast <= self.atMost):
            raise ValueError("atLeast and atMost must be in range (0, 4) and atLeast <= atMost.")

        transformations = [
        ('color', self.color_probability, color_ratio),
        ('artifact', self.artifact_probability, artifact_ratio),
        ('spatial_resolution', self.spatial_resolution_probability, spatial_resolution_ratio),
        ('noise', self.noise_probability, noise_ratio)
        ]

        transformations = [t for t in transformations if t[1] > 0]
        num_transformations = np.random.randint(self.atLeast, self.atMost +1)
        if (num_transformations > len(transformations)):
            num_transformations = len(transformations)

        if len(transformations) > num_transformations:
            probabilities = [t[1] for t in transformations]
            probabilities = np.array(probabilities) / np.sum(probabilities)
            selected_indices = np.random.choice(len(transformations), num_transformations, replace=False, p=probabilities)
            transformations = [transformations[i] for i in selected_indices]
    
        transformations.sort(key=lambda x: ['color', 'artifact', 'spatial_resolution', 'noise'].index(x[0]))
    
        for t in transformations:
            if t[0] == 'color':
                image = color_transformations.transform_color(image, reference_image, t[2], self.input_modality, self.output_modality)
            elif t[0] == 'artifact':
                image = artifact_transformations.transform_artifact(image, t[2], self.input_modality, self.output_modality)
            elif t[0] == 'spatial_resolution':
                image = spatial_resolution_transformations.transform_spatial_resolution(image, t[2], self.input_modality, self.output_modality)
            elif t[0] == 'noise':
                image = noise_transformations.transform_noise(image, t[2], self.input_modality, self.output_modality)
                                                              
        return image

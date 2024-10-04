from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality
from torchvision.transforms import v2
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid", palette="deep")



def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            image = np.load(file_path)
            images.append(image)
    return images

def load_augmented_images(folder_path, input_modality, output_modality, ratio):
    if ratio==0.0:
        return load_images(folder_path)
    images = []
    cross_modality_transformer = CrossModalityTransformations(
        input_modality=input_modality, 
        output_modality=output_modality, 
        atLeast=3, 
        atMost=4,
        color_probability=1.0, color_ratio_range=(ratio, ratio), 
        artifact_probability=1.0, artifact_ratio_range=(ratio, ratio), 
        spatial_resolution_probability=1.0, spatial_resolution_ratio_range=(ratio, ratio), 
        noise_probability=0, noise_ratio_range=(ratio, ratio)
        )
    transformer = v2.Compose([
        cross_modality_transformer.transform
    ])
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            image = np.load(file_path)
            image = transformer(image)
            images.append(image)
    return images

def run_experiment(input_modality, output_modality):
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Load the non-augmented images
    input_modality_images_path = "/mnt/data/jstutz/data/bladder_np_divided/" + input_modality.name + "/original"
    output_modality_images_path = "/mnt/data/jstutz/data/bladder_np_divided/" + output_modality.name + "/original"
    augmented_images_path = "/mnt/data/jstutz/data/bladder_np_divided/" + input_modality.name + "/to_be_augmented"
    real_images_modality_1 = load_images(input_modality_images_path)
    real_images_modality_2 = load_images(output_modality_images_path)

    for ratio in ratios:

        # Load the augmented images
        augmented_images_1_to_2 = load_augmented_images(augmented_images_path, input_modality, output_modality, ratio)

        # Combine the datasets
        combined_data = np.concatenate((real_images_modality_1, real_images_modality_2, augmented_images_1_to_2), axis=0)

        # Flatten each image (num_images, height, width) -> (num_images, height * width)
        num_images, height, width = combined_data.shape
        flattened_data = combined_data.reshape(num_images, height * width)

        # Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(flattened_data)

        # Create labels for the datasets
        labels = np.array([0]*len(real_images_modality_1) + [1]*len(real_images_modality_2) + [2]*len(augmented_images_1_to_2))

        # Apply PCA with 3 components
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(standardized_data)

        # Create a DataFrame for better handling with Seaborn
        df = pd.DataFrame({
            'Principal Component 1': principal_components[:, 0],
            'Principal Component 2': principal_components[:, 1],
            'Principal Component 3': principal_components[:, 2],
            'Dataset': labels
        })

        # Define custom names and markers
        dataset_names = [input_modality.name, output_modality.name, (input_modality.name + " to " + output_modality.name)]
        markers = ['o', 's', '^']

        # Plot the principal components in 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each dataset with different markers
        for i in range(3):
            subset = df[df['Dataset'] == i]
            ax.scatter(subset['Principal Component 1'], subset['Principal Component 2'], subset['Principal Component 3'],
                    label=dataset_names[i], marker=markers[i], alpha=0.7)

        # Customizing the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, dataset_names)

        plt.title('3D PCA of Medical Image Datasets')

        file_path = "averagePCA_" + input_modality.name + "_to_" + output_modality.name + "_" + str(ratio) + ".png"
        #file_path = "PCA_" + input_modality.name + "_to_" + output_modality.name + ".png"
    
        # Save the plot as a PNG file
        plt.savefig(file_path)
        plt.close()

input_modality = Input_Modality.PET
output_modality = Output_Modality.MRI
run_experiment(input_modality, output_modality)
input_modality = Input_Modality.PET
output_modality = Output_Modality.CT
run_experiment(input_modality, output_modality)
input_modality = Input_Modality.MRI
output_modality = Output_Modality.PET
run_experiment(input_modality, output_modality)
input_modality = Input_Modality.MRI
output_modality = Output_Modality.CT
run_experiment(input_modality, output_modality)
input_modality = Input_Modality.CT
output_modality = Output_Modality.PET
run_experiment(input_modality, output_modality)
input_modality = Input_Modality.CT
output_modality = Output_Modality.MRI
run_experiment(input_modality, output_modality)   
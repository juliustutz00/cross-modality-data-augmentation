from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



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
        transformation_probability=1.0,
        atLeast=2, 
        atMost=3,
        color_probability=0.95, color_ratio_range=(ratio, ratio), 
        artifact_probability=0.05, artifact_ratio_range=(ratio, ratio), 
        spatial_resolution_probability=0.8, spatial_resolution_ratio_range=(ratio, ratio), 
        noise_probability=0.2, noise_ratio_range=(ratio, ratio)
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

def calculate_glcm_features(image):
    distances = [1, 2, 3, 4, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = {
        'contrast': np.mean(graycoprops(glcm, 'contrast')),
        'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
        'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
        'energy': np.mean(graycoprops(glcm, 'energy')),
        'correlation': np.mean(graycoprops(glcm, 'correlation'))
    }
    
    return features

def calculate_average_glcm_features(dataset):
    all_features = [calculate_glcm_features(image) for image in dataset]
    avg_features = {key: np.mean([f[key] for f in all_features]) for key in all_features[0]}
    return avg_features

def run_experiment(input_modality, output_modality):
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Load the non-augmented images
    input_modality_images_path = "/mnt/data/jstutz/data/bladder_np_divided/" + input_modality.name + "/original"
    output_modality_images_path = "/mnt/data/jstutz/data/bladder_np_divided/" + output_modality.name + "/original"
    augmented_images_path = "/mnt/data/jstutz/data/bladder_np_divided/" + input_modality.name + "/to_be_augmented"
    real_images_modality_1 = load_images(input_modality_images_path)
    real_images_modality_2 = load_images(output_modality_images_path)

    distances_modality_1 = []
    distances_modality_2 = []

    for ratio in ratios:

        # Load the augmented images
        augmented_images_1_to_2 = load_augmented_images(augmented_images_path, input_modality, output_modality, ratio)

        # Calculate features
        real_features_modality_1 = calculate_average_glcm_features(real_images_modality_1)
        real_features_modality_2 = calculate_average_glcm_features(real_images_modality_2)
        augmented_features_1_to_2 = calculate_average_glcm_features(augmented_images_1_to_2)

        # Store features as vectors
        vector_modality_1 = list(real_features_modality_1.values())
        vector_modality_2 = list(real_features_modality_2.values())
        vector_augmented_1_to_2 = list(augmented_features_1_to_2.values())

        # Calculare distance between datasets
        # distance_modality_1 should get bigger the heavier the images are augmented
        # distance_modality_2 should get smaller the heavier the images are augmented
        #distance_modality_1 = euclidean(vector_modality_1, vector_augmented_1_to_2)
        #distance_modality_2 = euclidean(vector_modality_2, vector_augmented_1_to_2)
        scaler = StandardScaler()
        # Combine vectors to fit scaler
        combined_vectors = np.vstack([vector_modality_1, vector_augmented_1_to_2, vector_modality_2])
        # Fit and transform the vectors
        scaled_vectors = scaler.fit_transform(combined_vectors)
        # Extract scaled vectors
        scaled_vector_modality_1 = scaled_vectors[0]
        scaled_vector_augmented_1_to_2 = scaled_vectors[1]
        scaled_vector_modality_2 = scaled_vectors[2]
        # Calculate distances
        distance_modality_1 = euclidean(scaled_vector_modality_1, scaled_vector_augmented_1_to_2)
        distance_modality_2 = euclidean(scaled_vector_modality_2, scaled_vector_augmented_1_to_2)

        distances_modality_1.append(distance_modality_1)
        distances_modality_2.append(distance_modality_2)
        
        result_file.write("Ratio %f: \r\n" % (ratio))
        result_file.write("GLCM difference to input modality: %f \r\n" % (distance_modality_1))
        result_file.write("\r\n")
        result_file.write("GLCM difference to output modality: %f \r\n" % (distance_modality_2))
        result_file.write("\r\n")
    
    # Create a DataFrame for plotting
    data = {
        'Ratio': ratios + ratios,
        'Distance': distances_modality_1 + distances_modality_2,
        'Type': ['Input Modality'] * len(ratios) + ['Output Modality'] * len(ratios)
    }
    df = pd.DataFrame(data)

    # Plot using seaborn
    sns.set_theme(style="darkgrid", palette="deep")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Ratio', y='Distance', hue='Type', marker='o')
    plt.title('GLCM Feature Distances vs. Augmentation Ratio')
    plt.xlabel('Ratio')
    plt.ylabel('Distance')
    file_path = "glcm_features_" + input_modality.name + "_to_" + output_modality.name + ".png"
    
    # Save the plot as a PNG file
    plt.savefig(file_path)
    plt.close()


with open("glcm_feature_difference.txt", "w+", encoding="utf-8") as result_file:
    result_file.write("PET to MRI \r\n")
    input_modality = Input_Modality.PET
    output_modality = Output_Modality.MRI
    run_experiment(input_modality, output_modality)
    result_file.write("\r\n")
    result_file.write("PET to CT \r\n")
    input_modality = Input_Modality.PET
    output_modality = Output_Modality.CT
    run_experiment(input_modality, output_modality)
    result_file.write("\r\n")
    result_file.write("MRI to PET \r\n")
    input_modality = Input_Modality.MRI
    output_modality = Output_Modality.PET
    run_experiment(input_modality, output_modality)
    result_file.write("\r\n")
    result_file.write("MRI to CT \r\n")
    input_modality = Input_Modality.MRI
    output_modality = Output_Modality.CT
    run_experiment(input_modality, output_modality)
    result_file.write("\r\n")
    result_file.write("CT to PET \r\n")
    input_modality = Input_Modality.CT
    output_modality = Output_Modality.PET
    run_experiment(input_modality, output_modality)
    result_file.write("\r\n")
    result_file.write("CT to MRI \r\n")
    input_modality = Input_Modality.CT
    output_modality = Output_Modality.MRI
    run_experiment(input_modality, output_modality)
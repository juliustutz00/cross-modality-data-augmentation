from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality
import torch
import torchvision.models as models
import torchvision.transforms.v2 as v2
import numpy as np
import random
import pandas as pd
from scipy.linalg import sqrtm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
current_seed = 42
np.random.seed(current_seed)
random.seed(current_seed)
warnings.filterwarnings("ignore")

def load_images(folder_path):
    images = []
    transformer = v2.Compose([
        v2.Resize(299, 299), 
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            image = np.load(file_path)
            image = transformer(image)
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
        cross_modality_transformer.transform,
        v2.Resize(299, 299), 
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            image = np.load(file_path)
            image = transformer(image)
            images.append(image)
    return images

def get_features(model, images):
    model.eval()
    all_features = []
    with torch.no_grad():
        for image in images:
            image = image.unsqueeze(0)
            features = model(image)
            all_features.append(features.numpy().flatten())
    return np.array(all_features)

def calculate_fid(features_real, features_gen):
    # Calculate mean and covariance matrix
    mu_real, sigma_real = np.mean(features_real, axis=0), np.cov(features_real, rowvar=False)
    mu_gen, sigma_gen = np.mean(features_gen, axis=0), np.cov(features_gen, rowvar=False)
    
    # Calculate Fréchet distance
    ssdiff = np.sum((mu_real - mu_gen)**2.0)
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return fid

def run_experiment(input_modality, output_modality):
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fid_values_modality_1 = []
    fid_values_modality_2 = []

    for ratio in ratios:
    
        # Define model
        model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        model.fc = torch.nn.Identity()

        # Load the images
        input_modality_images_path = "path/to/your/data" + input_modality.name + "/original"
        output_modality_images_path = "path/to/your/data" + output_modality.name + "/original"
        augmented_images_path = "path/to/your/data" + input_modality.name + "/to_be_augmented"
        real_images_modality_1 = load_images(input_modality_images_path)
        real_images_modality_2 = load_images(output_modality_images_path)
        augmented_images_1_to_2 = load_augmented_images(augmented_images_path, input_modality, output_modality, ratio)

        # Calculate features
        real_features_modality_1 = get_features(model, real_images_modality_1)
        real_features_modality_2 = get_features(model, real_images_modality_2)
        augmented_features_1_to_2 = get_features(model, augmented_images_1_to_2)

        # Calculate FID
        # fid_modality_1 should get bigger the heavier the images are augmented
        # fid_modality_1 should get smaller the heavier the images are augmented
        fid_modality_1 = calculate_fid(real_features_modality_1, augmented_features_1_to_2)
        fid_modality_2 = calculate_fid(real_features_modality_2, augmented_features_1_to_2)

        fid_values_modality_1.append(fid_modality_1)
        fid_values_modality_2.append(fid_modality_2)
        
        result_file.write("Ratio %f: \r\n" % (ratio))
        result_file.write("FID to input modality: %f \r\n" % (fid_modality_1))
        result_file.write("FID to output modality: %f \r\n" % (fid_modality_2))

    # Create a DataFrame for plotting
    data = {
        'Ratio': ratios + ratios,
        'FID': fid_values_modality_1 + fid_values_modality_2,
        'Type': ['Input Modality'] * len(ratios) + ['Output Modality'] * len(ratios)
    }
    df = pd.DataFrame(data)

    # Plot using seaborn
    sns.set_theme(style="darkgrid", palette="deep")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Ratio', y='FID', hue='Type', marker='o')
    plt.title('FID vs. Augmentation Ratio')
    plt.xlabel('Ratio')
    plt.ylabel('FID')
    file_path = "fid_" + input_modality.name + "_to_" + output_modality.name + ".png"
    
    # Save the plot as a PNG file
    plt.savefig(file_path)
    plt.close()


with open("frechet_inception_distance.txt", "w+", encoding="utf-8") as result_file:
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
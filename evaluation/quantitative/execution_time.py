from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality
from torchvision.transforms import v2
from imgaug import augmenters as iaa
import albumentations as A
import time
import os
import numpy as np
import torch
from torchvision.transforms import v2

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            image = np.load(file_path)
            images.append(image)
    return images

def measure_time(func, data, *args):
    times = []
    for _ in range(1000):
        start = time.time()
        func(data, *args)
        end = time.time()
        times.append(end - start)
    return times

def run_imgaug(data):
    transformer = iaa.Sequential([
        iaa.Sometimes(0.1, iaa.Fliplr(1.0)),
        iaa.Sometimes(0.1, iaa.Affine(rotate=(-10, 10))),
        iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 1.0))),
        iaa.Sometimes(0.1, iaa.Multiply((0.9, 1.1))),
        iaa.Sometimes(0.1, iaa.LinearContrast((0.9, 1.1))), 
        iaa.Sometimes(0.1, iaa.Crop(percent=(0, 0.05)))
    ])
    times = measure_time(lambda data: [transformer.augment_image(img) for img in data], data)
    return np.mean(times), np.std(times)

def run_albumentations(data):
    transformer = A.Compose([
        A.HorizontalFlip(p=0.1),
        A.Rotate(limit=10, p=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
        A.RandomCrop(height=256, width=256, p=0.1)
    ])
    times = measure_time(lambda data: [transformer(image=image)['image'] for image in data], data)
    return np.mean(times), np.std(times)

def run_v2(data):
    transformer = v2.Compose([
        v2.RandomHorizontalFlip(p=0.1),
        v2.RandomApply([v2.RandomRotation(10)], p=0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))], p=0.1),
        v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.1),
        v2.RandomApply([v2.RandomResizedCrop(size=256, scale=(0.95, 1.0))], p=0.1)
    ])
    tensor_transformer = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])
    data_tensor = [tensor_transformer(img) for img in data]
    times = measure_time(lambda data_tensor: [transformer(img) for img in data_tensor], data_tensor)
    return np.mean(times), np.std(times)

def run_randaug(data):
    transformer = v2.Compose([
        v2.RandAugment()
    ])
    tensor_transformer = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])
    data_tensor = [tensor_transformer(img) for img in data]
    times = measure_time(lambda data_tensor: [transformer(img) for img in data_tensor], data_tensor)
    return np.mean(times), np.std(times)

def run_cross_modality(data, input_modality, output_modality):
    transformer = CrossModalityTransformations(
        input_modality=input_modality, 
        output_modality=output_modality, 
        transformation_probability=0.1,
        atLeast=0, 
        atMost=4,
        color_probability=0.9, color_ratio_range=(0, 1), 
        artifact_probability=0.05, artifact_ratio_range=(0, 1), 
        spatial_resolution_probability=0.6, spatial_resolution_ratio_range=(0, 1), 
        noise_probability=0.2, noise_ratio_range=(0, 1)
    )
    times = measure_time(lambda data: [transformer.transform(img) for img in data], data)
    return np.mean(times), np.std(times)

def run_experiment():
    data_PET = load_images("/mnt/data/jstutz/data/brain_np/PET/both")
    data_MRI = load_images("/mnt/data/jstutz/data/brain_np/MRI/both")
    data_CT = load_images("/mnt/data/jstutz/data/brain_np/CT/both")

    with open("execution_time.txt", "w+", encoding="utf-8") as result_file:
        # PET data
        result_file.write("PET dataset:\n")
        mean_time, std_time = run_imgaug(data_PET)
        result_file.write(f"imgaug: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_albumentations(data_PET)
        result_file.write(f"Albumentations: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_v2(data_PET)
        result_file.write(f"v2: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_randaug(data_PET)
        result_file.write(f"RandAugment: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_cross_modality(data_PET, Input_Modality.PET, Output_Modality.MRI)
        result_file.write(f"cross_modality to MRI: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_cross_modality(data_PET, Input_Modality.PET, Output_Modality.CT)
        result_file.write(f"cross_modality to CT: mean={mean_time:.6f}, std={std_time:.6f}\n")
        result_file.write("\n")

        # MRI data
        result_file.write("MRI dataset:\n")
        mean_time, std_time = run_imgaug(data_MRI)
        result_file.write(f"imgaug: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_albumentations(data_MRI)
        result_file.write(f"Albumentations: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_v2(data_MRI)
        result_file.write(f"v2: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_randaug(data_MRI)
        result_file.write(f"RandAugment: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_cross_modality(data_MRI, Input_Modality.MRI, Output_Modality.PET)
        result_file.write(f"cross_modality to PET: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_cross_modality(data_MRI, Input_Modality.MRI, Output_Modality.CT)
        result_file.write(f"cross_modality to CT: mean={mean_time:.6f}, std={std_time:.6f}\n")
        result_file.write("\n")

        # CT data
        result_file.write("CT dataset:\n")
        mean_time, std_time = run_imgaug(data_CT)
        result_file.write(f"imgaug: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_albumentations(data_CT)
        result_file.write(f"Albumentations: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_v2(data_CT)
        result_file.write(f"v2: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_randaug(data_CT)
        result_file.write(f"RandAugment: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_cross_modality(data_CT, Input_Modality.CT, Output_Modality.PET)
        result_file.write(f"cross_modality to PET: mean={mean_time:.6f}, std={std_time:.6f}\n")
        mean_time, std_time = run_cross_modality(data_CT, Input_Modality.CT, Output_Modality.MRI)
        result_file.write(f"cross_modality to MRI: mean={mean_time:.6f}, std={std_time:.6f}\n")
        result_file.write("\n")

run_experiment()

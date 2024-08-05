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

def run_imgaug(data):
    transformer = iaa.Sequential([
            iaa.Sometimes(0.1, iaa.Fliplr(1.0)),
            iaa.Sometimes(0.1, iaa.Affine(rotate=(-10, 10))),
            iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Sometimes(0.1, iaa.Multiply((0.9, 1.1))),
            iaa.Sometimes(0.1, iaa.LinearContrast((0.9, 1.1))), 
            iaa.Sometimes(0.1, iaa.Crop(percent=(0, 0.05)))
        ])
    start = time.time()
    for image in data:
        transformer.augment_image(image)
    end = time.time()
    result_file.write("imgaug: %f\r\n" % (end - start))

def run_albumentations(data):
    transformer = A.Compose([
            A.HorizontalFlip(p=0.1),
            A.Rotate(limit=10, p=0.1),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
            A.RandomCrop(height=256, width=256, p=0.1)
        ])
    start = time.time()
    for image in data:
        transformer(image=image)['image']
    end = time.time()
    result_file.write("Albumentations: %f\r\n" % (end - start))

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
    data_tensor = []
    for image in data:
        image_tensor = tensor_transformer(image)
        data_tensor.append(image_tensor)
    start = time.time()
    for image in data_tensor:
        transformer(image)
    end = time.time()
    result_file.write("v2: %f\r\n" % (end - start))

def run_randaug(data):
    transformer = v2.Compose([
                v2.RandAugment()
                ])
    tensor_transformer = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])
    data_tensor = []
    for image in data:
        image_tensor = tensor_transformer(image)
        data_tensor.append(image_tensor)
    start = time.time()
    for image in data_tensor:
        transformer(image)
    end = time.time()
    result_file.write("RandAugment: %f\r\n" % (end - start))          

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
    start = time.time()
    for image in data:
        transformer.transform(image)
    end = time.time()
    result_file.write("cross_modality to %s: %f\r\n" % (output_modality.name, (end - start)))


def run_experiment():
    
    data_PET = load_images("path/to/your/data")
    data_MRI = load_images("path/to/your/data")
    data_CT = load_images("path/to/your/data")
    
    # PET data
    result_file.write("PET dataset: \r\n")
    run_imgaug(data_PET)
    run_albumentations(data_PET)
    run_v2(data_PET)
    run_randaug(data_PET)
    run_cross_modality(data_PET, Input_Modality.PET, Output_Modality.MRI)
    run_cross_modality(data_PET, Input_Modality.PET, Output_Modality.CT)
    result_file.write("\r\n")

    # MRI data
    result_file.write("MRI dataset: \r\n")
    run_imgaug(data_MRI)
    run_albumentations(data_MRI)
    run_v2(data_MRI)
    run_randaug(data_MRI)
    run_cross_modality(data_MRI, Input_Modality.MRI, Output_Modality.PET)
    run_cross_modality(data_MRI, Input_Modality.MRI, Output_Modality.CT)
    result_file.write("\r\n")
    
    # CT data
    result_file.write("CT dataset: \r\n")
    run_imgaug(data_CT)
    run_albumentations(data_CT)
    run_v2(data_CT)
    run_randaug(data_CT)
    run_cross_modality(data_CT, Input_Modality.CT, Output_Modality.PET)
    run_cross_modality(data_CT, Input_Modality.CT, Output_Modality.MRI)
    result_file.write("\r\n")


with open("execution_time.txt", "w+", encoding="utf-8") as result_file:
    run_experiment()    
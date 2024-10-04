from classification_performance_utility import plot_and_save_roc_curve
from classification_performance_model_utility import train_resnet, train_vit, validate, evaluate
from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, vit_b_16, ViT_B_16_Weights
from torchvision.transforms import v2, functional
from imgaug import augmenters as iaa
import albumentations as A
from pytorch_ood.detector import EnergyBased
from pytorch_ood.utils import OODMetrics
import warnings
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
current_seed = 42
np.random.seed(current_seed)
random.seed(current_seed)
torch.manual_seed(current_seed)
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed_all(current_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")



class MedicalImageDataset(Dataset):
    def __init__(self, data, labels, model_name, transform=None, cross_modality=False, mean=None, std=None):
        self.data = data
        self.labels = labels
        self.model_name = model_name
        self.transform = transform
        self.cross_modality = cross_modality
        self.mean=mean
        self.std=std
        self.indices = [i for i in range(len(data))]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        image = self.data[i]
        label = self.labels[i]
        
        if self.transform and self.cross_modality==False:
            image = self.transform(image)
        elif self.cross_modality==True:
            transform = self.choose_transformation("MRI")
            image = transform(image)
        
        return image, label

    def choose_transformation(self, modality):
        if modality=='PET':
            x = Input_Modality.PET
            y = random.choice([Output_Modality.MRI, Output_Modality.CT])
        elif modality=='MRI':
            x = Input_Modality.MRI
            y = Output_Modality.PET
        elif modality=='CT':
            x = Input_Modality.CT
            y = random.choice([Output_Modality.PET, Output_Modality.MRI])
        else:
            raise ValueError("Input modality not implemented.")

        cross_modality_transformer = CrossModalityTransformations(
        input_modality=x, 
        output_modality=y, 
        atLeast=2, 
        atMost=3,
        color_probability=0.95, color_ratio_range=(0.75, 1), 
        artifact_probability=0.05, artifact_ratio_range=(0, 1), 
        spatial_resolution_probability=0.6, spatial_resolution_ratio_range=(0.75, 1), 
        noise_probability=0.2, noise_ratio_range=(0, 1)
        )

        if self.model_name == "ResNet":
            return v2.Compose([
                cross_modality_transformer.transform,
                v2.ToTensor(),
                v2.Normalize(mean=[self.mean], std=[self.std])
                ])
        elif self.model_name == "VisionTransformer":
            return v2.Compose([
                cross_modality_transformer.transform,
                v2.ToTensor(),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[self.mean, self.mean, self.mean], std=[self.std, self.std, self.std])
                ])
        else: 
            raise ValueError("Model not implemented.")

class ImgAugTransformResNet:
    def __init__(self, mean, std):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.1, iaa.Fliplr(1.0)),
            iaa.Sometimes(0.1, iaa.Affine(rotate=(-10, 10))),
            iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Sometimes(0.1, iaa.Multiply((0.9, 1.1))),
            iaa.Sometimes(0.1, iaa.LinearContrast((0.9, 1.1))), 
            iaa.Sometimes(0.1, iaa.Crop(percent=(0, 0.05)))
        ])
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = self.aug.augment_image(img)
        img = functional.to_tensor(img)
        img = functional.normalize(img, mean=self.mean, std=self.std)
        return img

class ImgAugTransformVisTran:
    def __init__(self, mean, std):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.1, iaa.Fliplr(1.0)),
            iaa.Sometimes(0.1, iaa.Affine(rotate=(-20, 20))),
            iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.1, iaa.Multiply((0.8, 1.2))),
            iaa.Sometimes(0.1, iaa.LinearContrast((0.75, 1.5))), 
            iaa.Sometimes(0.1, iaa.Crop(percent=(0, 0.1)))
        ])
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = self.aug.augment_image(img)
        img = functional.to_tensor(img)
        img = img.repeat(3, 1, 1)
        img = functional.resize(img, (224, 224))
        img = functional.normalize(img, mean=self.mean, std=self.std)
        return img

class AlbumentationsTransformResNet:
    def __init__(self, mean, std):
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.1),
            A.Rotate(limit=10, p=0.1),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
            A.RandomCrop(height=256, width=256, p=0.1),
        ])
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = self.aug(image=img)['image']
        img = functional.to_tensor(img)
        img = functional.normalize(img, mean=self.mean, std=self.std)
        return img

class AlbumentationsTransformVisTran:
    def __init__(self, mean, std):
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.1),
            A.Rotate(limit=10, p=0.1),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
            A.RandomCrop(height=256, width=256, p=0.1)
        ])
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = self.aug(image=img)['image']
        img = functional.to_tensor(img)
        img = img.repeat(3, 1, 1)
        img = functional.resize(img, (224, 224))
        img = functional.normalize(img, mean=self.mean, std=self.std)
        return img


def load_data(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            image = np.load(file_path)
            images.append(image)
    labels = torch.zeros(len(images), dtype=torch.long)
    return images, labels

def calculate_mean_std(data):
    return (np.mean(data) / 255.0), (np.std(data) / 255.0)

def define_transformations_no(model, x_set, mean, std):
    if (model.__class__.__name__ == "ResNet"):
        if (x_set == "train"):
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[mean], std=[std])
                ])
        elif (x_set == "val_test"): 
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[mean], std=[std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    elif (model.__class__.__name__ == "VisionTransformer"):
        if (x_set == "train"):
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        elif (x_set == "val_test"):
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    else:
        raise ValueError("Transformations for this model not implemented")

def define_transformations_imgaug(model, x_set, mean, std):
    if (model.__class__.__name__ == "ResNet"):
        if (x_set == "train"):
            return ImgAugTransformResNet(mean, std)
        elif (x_set == "val_test"): 
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[mean], std=[std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    elif (model.__class__.__name__ == "VisionTransformer"):
        if (x_set == "train"):
            return ImgAugTransformVisTran(mean, std)
        elif (x_set == "val_test"):
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    else:
        raise ValueError("Transformations for this model not implemented")

def define_transformations_albumentations(model, x_set, mean, std):
    if (model.__class__.__name__ == "ResNet"):
        if (x_set == "train"):
            return AlbumentationsTransformResNet(mean, std)
        elif (x_set == "val_test"): 
            return v2.Compose([
                v2.ToTensor(),
                v2.Normalize(mean=[mean], std=[std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    elif (model.__class__.__name__ == "VisionTransformer"):
        if (x_set == "train"):
            return AlbumentationsTransformVisTran(mean, std)
        elif (x_set == "val_test"):
            return v2.Compose([
                v2.ToTensor(),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    else:
        raise ValueError("Transformations for this model not implemented")

def define_transformations_v2(model, x_set, mean, std):
    if (model.__class__.__name__ == "ResNet"):
        if (x_set == "train"):
            return v2.Compose([
                v2.ToTensor(),
                v2.RandomHorizontalFlip(p=0.1),
                v2.RandomApply([v2.RandomRotation(10)], p=0.1),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))], p=0.1),
                v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.1),
                v2.RandomApply([v2.RandomResizedCrop(size=256, scale=(0.95, 1.0))], p=0.1),
                v2.Normalize(mean=[mean], std=[std])
                ])
        elif (x_set == "val_test"): 
            return v2.Compose([
                v2.ToTensor(),
                v2.Normalize(mean=[mean], std=[std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    elif (model.__class__.__name__ == "VisionTransformer"):
        if (x_set == "train"):
            return v2.Compose([
                v2.ToTensor(),
                v2.RandomHorizontalFlip(p=0.1),
                v2.RandomApply([v2.RandomRotation(10)], p=0.1),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))], p=0.1),
                v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.1),
                v2.RandomApply([v2.RandomResizedCrop(size=256, scale=(0.95, 1.0))], p=0.1),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        elif (x_set == "val_test"):
            return v2.Compose([
                v2.ToTensor(),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    else:
        raise ValueError("Transformations for this model not implemented")

def define_transformations_RandAugment(model, x_set, mean, std):
    if (model.__class__.__name__ == "ResNet"):
        if (x_set == "train"):
            return v2.Compose([
                v2.ToTensor(),
                v2.RandAugment(),
                v2.Normalize(mean=[mean], std=[std])
                ])
        elif (x_set == "val_test"): 
            return v2.Compose([
                v2.ToTensor(),
                v2.Normalize(mean=[mean], std=[std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    elif (model.__class__.__name__ == "VisionTransformer"):
        if (x_set == "train"):
            return v2.Compose([
                v2.ToTensor(),
                v2.RandAugment(),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        elif (x_set == "val_test"):
            return v2.Compose([
                v2.ToTensor(),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    else:
        raise ValueError("Transformations for this model not implemented")

def define_transformations_cross_modality(model, x_set, mean, std):
    if (model.__class__.__name__ == "ResNet"):
        if (x_set == "train"):
            return None
        elif (x_set == "val_test"): 
            return v2.Compose([
                v2.ToTensor(),
                v2.Normalize(mean=[mean], std=[std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    elif (model.__class__.__name__ == "VisionTransformer"):
        if (x_set == "train"):
            return None
        elif (x_set == "val_test"):
            return v2.Compose([
                v2.ToTensor(),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)), 
                v2.Resize((224, 224)),
                v2.Normalize(mean=[mean, mean, mean], std=[std, std, std])
                ])
        else:
            raise ValueError("Transformations only implemented for train / val / test set.")
    else:
        raise ValueError("Transformations for this model not implemented")

def define_transformations(model, x_set, mean, std, augmentation):
    if (augmentation == "no"):
        return define_transformations_no(model, x_set, mean, std)
    elif (augmentation == "imgaug"):
        return define_transformations_imgaug(model, x_set, mean, std)
    elif (augmentation == "albumentations"):
        return define_transformations_albumentations(model, x_set, mean, std)
    elif (augmentation == "v2"):
        return define_transformations_v2(model, x_set, mean, std)
    elif (augmentation == "RandAugment"):
        return define_transformations_RandAugment(model, x_set, mean, std)
    elif (augmentation == "cross_modality"):
        return define_transformations_cross_modality(model, x_set, mean, std)
    else:
        raise ValueError("Transformation for given augmentation not implemented.")

def stratified_split(data, labels, test_and_val_size, reduction_percentage):
    train_data, train_labels = [], []
    val_data, val_labels = [], []
    
    if reduction_percentage < 1.0:
        X_reduced, _, y_reduced, _ = train_test_split(data, labels, test_size=1-reduction_percentage, stratify=labels)
    else:
        X_reduced, y_reduced = data, labels
        
    # Perform stratified split
    X_train, X_val, y_train, y_val = train_test_split(X_reduced, y_reduced, test_size=test_and_val_size, stratify=y_reduced)
        
    train_data.extend(X_train)
    train_labels.extend(y_train)
    val_data.extend(X_val)
    val_labels.extend(y_val)
    
    return train_data, train_labels, val_data, val_labels


def run_experiment(runs, model, anatomy, augmentation, dataset_size=1.0):
    num_classes = 2
    balanced_acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_roc_list = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    AUROC_list = []
    AUPR_IN_list = []
    AUPR_OUT_list = []
    FPR95TPR_list = []

    #data, labels = load_data("/mnt/data/jstutz/data/brain_np_extended/train_val_set/MRI")
    #test_data_in, test_labels_in = load_data("/mnt/data/jstutz/data/brain_np_extended/test_set/MRI")
    #test_data_out, test_labels_out = load_data("/mnt/data/jstutz/data/brain_np_extended/test_set/CT")
    data, labels = load_data("/mnt/data/jstutz/data/bladder_np_divided/MRI/to_be_augmented")
    test_data_in, test_labels_in = load_data("/mnt/data/jstutz/data/bladder_np_divided/MRI/original")
    test_data_out, test_labels_out = load_data("/mnt/data/jstutz/data/bladder_np_divided/PET/original")

    test_labels_in = torch.zeros(len(test_data_in), dtype=torch.long)  # IN samples labeled as 0, cast to long
    test_labels_out = torch.ones(len(test_data_out), dtype=torch.long)  # OOD samples labeled as 1, cast to long
    
    # Calculate mean and std of dataset for normalization
    mean, std = calculate_mean_std(data)

    # Save original model state
    original_model_state = model.state_dict()

    for run in range(runs):
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Split and reduce the data
        train_data, train_labels, val_data, val_labels = stratified_split(data, labels, 0.17, dataset_size)

        # Create the datasets
        train_dataset = MedicalImageDataset(train_data, train_labels, model.__class__.__name__)
        val_dataset = MedicalImageDataset(val_data, val_labels, model.__class__.__name__)
        test_dataset_in = MedicalImageDataset(test_data_in, test_labels_in, model.__class__.__name__)
        test_dataset_out = MedicalImageDataset(test_data_out, test_labels_out, model.__class__.__name__)

        # Set special values if cross_modality augmentation is chosen
        if augmentation=="cross_modality":
            train_dataset.cross_modality=True
            train_dataset.mean = mean
            train_dataset.std = std

        # Define transformations based on model used
        train_transform = define_transformations(model, "train", mean, std, augmentation)
        val_test_transform = define_transformations(model, "val_test", mean, std, augmentation)

        train_dataset.transform = train_transform
        val_dataset.transform = val_test_transform
        test_dataset_in.transform = val_test_transform
        test_dataset_out.transform = val_test_transform

        # Handle batching
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset_in + test_dataset_out, batch_size=batch_size, shuffle=False)

        # Training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        num_epochs = 25
        best_val_acc = 0.0
        best_val_loss = np.inf
        best_model_weights = None

        for epoch in range(num_epochs):
            if (model.__class__.__name__ == "ResNet"):
                train_loss = train_resnet(model, train_loader, criterion, optimizer, device)
            elif (model.__class__.__name__ == "VisionTransformer"): 
                train_loss, train_acc = train_vit(model, train_loader, criterion, optimizer, device)
            
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Save the best model
            if val_acc > best_val_acc and val_loss <= best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model_weights = model.state_dict()

        model.load_state_dict(best_model_weights)

        with torch.no_grad():
            detector = EnergyBased(model)
            metrics = OODMetrics()
            for x, y in test_loader:
                metrics.update(detector(x.to(device)), y.to(device))
            final_metrics = metrics.compute()
            AUROC_list.append(final_metrics['AUROC'])
            AUPR_IN_list.append(final_metrics['AUPR-IN'])
            AUPR_OUT_list.append(final_metrics['AUPR-OUT'])
            FPR95TPR_list.append(final_metrics['FPR95TPR'])

        test_loss, test_acc, balanced_acc, precision, recall, f1, auc_roc, fpr, tpr, roc_auc = evaluate(model, test_loader, criterion, device)
        balanced_acc_list.append(balanced_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_roc_list.append(auc_roc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

        result_file.write("Run %d \r\n" % (run+1))
        result_file.write("AUROC: %f \r\n" % (final_metrics['AUROC']))
        result_file.write("AUPR-IN: %f \r\n" % (final_metrics['AUPR-IN']))
        result_file.write("AUPR-OUT: %f \r\n" % (final_metrics['AUPR-OUT']))
        result_file.write("FPR95TPR: %f \r\n" % (final_metrics['FPR95TPR']))

        # Reset model weights
        model.load_state_dict(original_model_state)
        


    average_balanced_acc = sum(balanced_acc_list) / len(balanced_acc_list)
    average_precision = sum(precision_list) / len(precision_list)
    average_recall = sum(recall_list) / len(recall_list)
    average_f1 = sum(f1_list) / len(f1_list)
    average_auc_roc = sum(auc_roc_list) / len(auc_roc_list)
    file_path = 'average_roc_curve_OOD_' + model.__class__.__name__ + '_' + augmentation + '.png'
    plot_and_save_roc_curve(fpr_list, tpr_list, roc_auc_list, file_path)

    average_AUROC = sum(AUROC_list) / len(AUROC_list)
    average_AUPR_IN = sum(AUPR_IN_list) / len(AUPR_IN_list)
    average_AUPR_OUT = sum(AUPR_OUT_list) / len(AUPR_OUT_list)
    average_FPR95TPR = sum(FPR95TPR_list) / len(FPR95TPR_list)

    result_file.write("\r\n")
    result_file.write("Average Test Metrics\r\n")
    result_file.write("AUROC: %f \r\n" % (average_AUROC))
    result_file.write("AUPR-IN: %f \r\n" % (average_AUPR_IN))
    result_file.write("AUPR-OUT: %f \r\n" % (average_AUPR_OUT))
    result_file.write("FPR95TPR: %f \r\n" % (average_FPR95TPR))
    result_file.write("\r\n")


with open("OOD_samples_MRI_to_PET.txt", "w+", encoding="utf-8") as result_file:
    runs = 5
    ood_insgesamt =[]


    result_file.write("ResNet \r\n")

    result_file.write("no \r\n")
    modelResNet_no = resnet18(pretrained=False, num_classes=2)
    modelResNet_no.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    run_experiment(runs, modelResNet_no, "brain", "no")

    result_file.write("imgaug \r\n")
    modelResNet_imgaug = resnet18(pretrained=False, num_classes=2)
    modelResNet_imgaug.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    run_experiment(runs, modelResNet_imgaug, "brain", "imgaug")

    result_file.write("albumentations \r\n")
    modelResNet_albumentations = resnet18(pretrained=False, num_classes=2)
    modelResNet_albumentations.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    run_experiment(runs, modelResNet_albumentations, "brain", "albumentations")

    result_file.write("v2 \r\n")
    modelResNet_v2 = resnet18(pretrained=False, num_classes=2)
    modelResNet_v2.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    run_experiment(runs, modelResNet_v2, "brain", "v2")
    
    result_file.write("RandAugment \r\n")
    modelResNet_RandAugment = resnet18(pretrained=False, num_classes=2)
    modelResNet_RandAugment.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    run_experiment(runs, modelResNet_RandAugment, "brain", "RandAugment")

    result_file.write("cross_modality \r\n")
    modelResNet_cross_modality = resnet18(pretrained=False, num_classes=2)
    modelResNet_cross_modality.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    run_experiment(runs, modelResNet_cross_modality, "brain", "cross_modality")

    
    
    result_file.write("\r\n")
    result_file.write("Vision Transformer \r\n")

    result_file.write("no \r\n")
    modelVisTran_no = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    num_ftrs_no = modelVisTran_no.heads.head.in_features
    for params in modelVisTran_no.parameters():
        params.requires_grad=False
    modelVisTran_no.heads.head = nn.Linear(num_ftrs_no, 2)
    run_experiment(runs, modelVisTran_no, "brain", "no")
    
    result_file.write("imgaug \r\n")
    modelVisTran_imgaug = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    num_ftrs_imgaug = modelVisTran_imgaug.heads.head.in_features
    for params in modelVisTran_imgaug.parameters():
        params.requires_grad=False
    modelVisTran_imgaug.heads.head = nn.Linear(num_ftrs_imgaug, 2)
    run_experiment(runs, modelVisTran_imgaug, "brain", "imgaug")

    result_file.write("albumentations \r\n")
    modelVisTran_albumentations = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    num_ftrs_albumentations = modelVisTran_albumentations.heads.head.in_features
    for params in modelVisTran_albumentations.parameters():
        params.requires_grad=False
    modelVisTran_albumentations.heads.head = nn.Linear(num_ftrs_albumentations, 2)
    run_experiment(runs, modelVisTran_albumentations, "brain", "albumentations")

    result_file.write("v2 \r\n")
    modelVisTran_v2 = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    num_ftrs_v2 = modelVisTran_v2.heads.head.in_features
    for params in modelVisTran_v2.parameters():
        params.requires_grad=False
    modelVisTran_v2.heads.head = nn.Linear(num_ftrs_v2, 2)
    run_experiment(runs, modelVisTran_v2, "brain", "v2")

    result_file.write("RandAugment \r\n")
    modelVisTran_RandAugment = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    num_ftrs_RandAugment = modelVisTran_RandAugment.heads.head.in_features
    for params in modelVisTran_RandAugment.parameters():
        params.requires_grad=False
    modelVisTran_RandAugment.heads.head = nn.Linear(num_ftrs_RandAugment, 2)
    run_experiment(runs, modelVisTran_RandAugment, "brain", "RandAugment")

    result_file.write("cross_modality \r\n")
    modelVisTran_cross_modality = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    num_ftrs_cross_modality = modelVisTran_cross_modality.heads.head.in_features
    for params in modelVisTran_cross_modality.parameters():
        params.requires_grad=False
    modelVisTran_cross_modality.heads.head = nn.Linear(num_ftrs_cross_modality, 2)
    run_experiment(runs, modelVisTran_cross_modality, "brain", "cross_modality")
    
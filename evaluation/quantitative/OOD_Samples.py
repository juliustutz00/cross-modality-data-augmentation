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
from torchvision.transforms import v2
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
        atLeast=0, 
        atMost=4,
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


def load_data(root_dir):
    class_1 = "healthy"
    class_2 = "not_healthy"
    
    data = []
    labels = []
    label_map = {class_1: 0, class_2: 1}
    
    for label in [class_1, class_2]:
        folder_path = os.path.join(root_dir, label)
        for filename in os.listdir(folder_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(folder_path, filename)
                image = np.load(file_path)
                data.append(image)
                labels.append(label_map[label])
    
    return data, labels

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
    ood_list = []

    data, labels = load_data("path/to/your/data")
    test_data, test_labels = load_data("path/to/your/data")

    # Calculate mean and std of dataset for normalization
    mean, std = calculate_mean_std(data)

    # Save original model state
    original_model_state = model.state_dict()

    for run in range(runs):

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

       # Split and reduce the data
        train_data, train_labels, val_data, val_labels = stratified_split(data, labels, 0.1, dataset_size)

        # Create the datasets
        train_dataset = MedicalImageDataset(train_data, train_labels, model.__class__.__name__)
        val_dataset = MedicalImageDataset(val_data, val_labels, model.__class__.__name__)
        test_dataset = MedicalImageDataset(test_data, test_labels, model.__class__.__name__)

        # Set special values if cross_modality augmentation is chosen
        if augmentation=="cross_modality":
            train_dataset.cross_modality=True
            train_dataset.mean = mean
            train_dataset.std = std

        # Handle batching
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        # Define transformations based on model used
        train_transform = define_transformations(model, "train", mean, std, augmentation)
        val_test_transform = define_transformations(model, "val_test", mean, std, augmentation)

        train_dataset.transform = train_transform
        val_dataset.transform = val_test_transform
        test_dataset.transform = val_test_transform

        # Training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        num_epochs = 25
        best_val_acc = 0.0
        best_val_loss = np.inf
        best_train_loss = np.inf
        best_model_weights = None

        for epoch in range(num_epochs):
            if (model.__class__.__name__ == "ResNet"):
                train_loss = train_resnet(model, train_loader, criterion, optimizer, device)
            elif (model.__class__.__name__ == "VisionTransformer"): 
                train_loss, train_acc = train_vit(model, train_loader, criterion, optimizer, device)
            
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Save the best model
            if val_acc > best_val_acc and val_loss <= best_val_loss and train_loss <= best_train_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_model_weights = model.state_dict()

        model.load_state_dict(best_model_weights)

        model.eval()
        ood_scores = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                softmax_probs = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, _ = torch.max(softmax_probs, dim=1)
                ood_scores.extend(max_probs.cpu().numpy())

        threshold = 0.999
        ood_samples = [score for score in ood_scores if score < threshold]

        test_loss, test_acc, balanced_acc, precision, recall, f1, auc_roc, fpr, tpr, roc_auc = evaluate(model, test_loader, criterion, device)
        balanced_acc_list.append(balanced_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_roc_list.append(auc_roc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
        ood_list.append(ood_scores)

        result_file.write("Run %d \r\n" % (run+1))
        result_file.write("Balanced Accuracy: %f \r\n" % (balanced_acc))
        result_file.write("Precision: %f \r\n" % (precision))
        result_file.write("Recall: %f \r\n" % (recall))
        result_file.write("F1-Score: %f \r\n" % (f1))
        result_file.write("AUC-ROC: %f \r\n" % (auc_roc))

        # Reset model weights
        model.load_state_dict(original_model_state)
        


    average_balanced_acc = sum(balanced_acc_list) / len(balanced_acc_list)
    average_precision = sum(precision_list) / len(precision_list)
    average_recall = sum(recall_list) / len(recall_list)
    average_f1 = sum(f1_list) / len(f1_list)
    average_auc_roc = sum(auc_roc_list) / len(auc_roc_list)
    ood_insgesamt.append(ood_list)
    file_path = 'average_roc_curve_OOD_' + model.__class__.__name__ + '_' + augmentation + '.png'
    plot_and_save_roc_curve(fpr_list, tpr_list, roc_auc_list, file_path)

    result_file.write("\r\n")
    result_file.write("Final Average Test Metrics \r\n")
    result_file.write("Balanced Accuracy: %f \r\n" % (average_balanced_acc))
    result_file.write("Precision: %f \r\n" % (average_precision))
    result_file.write("Recall: %f \r\n" % (average_recall))
    result_file.write("F1-Score: %f \r\n" % (average_f1))
    result_file.write("AUC-ROC: %f \r\n" % (average_auc_roc))
    result_file.write("\r\n")


with open("OOD_samples.txt", "w+", encoding="utf-8") as result_file:
    runs = 5
    ood_insgesamt =[]


    result_file.write("ResNet \r\n")

    result_file.write("no \r\n")
    modelResNet_no = resnet18(pretrained=False, num_classes=2)
    modelResNet_no.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    run_experiment(runs, modelResNet_no, "brain", "no")

    result_file.write("cross_modality \r\n")
    modelResNet_cross_modality = resnet18(pretrained=False, num_classes=2)
    modelResNet_cross_modality.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    run_experiment(runs, modelResNet_cross_modality, "brain", "cross_modality")

    result = [[] for _ in range(5)]
    # Calculate the difference between the softmax values calculated for the non-augmented and the cross modality augmented dataset
    for i in range(5):
        for j in range(48):
            diff = ood_insgesamt[1][i][j] - ood_insgesamt[0][i][j]
            result[i].append(diff)

    result = np.array(result)
    mean_result = np.mean(result, axis=0)
    result_file.write("\r\n")
    for value in mean_result:
        result_file.write(f"{value},\n")
    
    
    result_file.write("\r\n")
    result_file.write("Vision Transformer \r\n")

    result_file.write("no \r\n")
    modelVisTran_no = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    num_ftrs_no = modelVisTran_no.heads.head.in_features
    for params in modelVisTran_no.parameters():
        params.requires_grad=False
    modelVisTran_no.heads.head = nn.Linear(num_ftrs_no, 2)
    run_experiment(runs, modelVisTran_no, "brain", "no")
    
    result_file.write("cross_modality \r\n")
    modelVisTran_cross_modality = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    num_ftrs_cross_modality = modelVisTran_cross_modality.heads.head.in_features
    for params in modelVisTran_cross_modality.parameters():
        params.requires_grad=False
    modelVisTran_cross_modality.heads.head = nn.Linear(num_ftrs_cross_modality, 2)
    run_experiment(runs, modelVisTran_cross_modality, "brain", "cross_modality")

    result = [[] for _ in range(5)]
    # Calculate the difference between the softmax values calculated for the non-augmented and the cross modality augmented dataset
    for i in range(5):
        for j in range(48):
            diff = ood_insgesamt[3][i][j] - ood_insgesamt[2][i][j]
            result[i].append(diff)

    result = np.array(result)
    mean_result = np.mean(result, axis=0)
    result_file.write("\r\n")
    for value in mean_result:
        result_file.write(f"{value},\n")
    
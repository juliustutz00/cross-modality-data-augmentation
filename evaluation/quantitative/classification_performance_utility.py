import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns



def print_dataset_distribution(data, labels):
    distribution = defaultdict(int)
    for modality in data:
        for label in labels[modality]:
            distribution[(modality, label)] += 1
    for key, count in distribution.items():
        modality, label = key
        label_name = 'healthy' if label == 0 else 'not_healthy'
        print(f"Modality: {modality}, Label: {label_name}, Count: {count}")

def calculate_mean_std(data):
    dataCT = data["CT"]
    dataMRI = data["MRI"]
    dataPET = data["PET"]
    dataCT = np.concatenate(dataCT, axis=0)  # Flatten the list of arrays
    meanCT = np.mean(dataCT) / 255.0
    stdCT = np.std(dataCT) / 255.0
    dataMRI = np.concatenate(dataMRI, axis=0)  # Flatten the list of arrays
    meanMRI = np.mean(dataMRI) / 255.0
    stdMRI = np.std(dataMRI) / 255.0
    dataPET = np.concatenate(dataPET, axis=0)  # Flatten the list of arrays
    meanPET = np.mean(dataPET) / 255.0
    stdPET = np.std(dataPET) / 255.0

    mean = (meanCT + meanMRI + meanPET) / 3
    std = (stdCT + stdMRI + stdPET) / 3
    return mean, std

# Function to load data
def load_data(root_dir, anatomy):
    if (anatomy == "brain"):
        class_1 = "healthy"
        class_2 = "not_healthy"
    elif (anatomy == "bladder"):
        class_1 = "stage_II"
        class_2 = "stage_III"
    else:
        raise ValueError("This anatomy is not implemented.")
    
    data = defaultdict(list)
    labels = defaultdict(list)
    label_map = {class_1: 0, class_2: 1}
    
    for modality in ['CT', 'MRI', 'PET']:
        for label in [class_1, class_2]:
            folder_path = os.path.join(root_dir, modality, label)
            for filename in os.listdir(folder_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(folder_path, filename)
                    image = np.load(file_path)
                    data[modality].append(image)
                    labels[modality].append(label_map[label])
    
    return data, labels

# Split and reduce the dataset into train / validate / test set
def stratified_split(data, labels, test_and_val_size, reduction_percentage):
    train_data, train_labels = defaultdict(list), defaultdict(list)
    val_data, val_labels = defaultdict(list), defaultdict(list)
    
    for modality in data:
        # Reduce the dataset size
        if reduction_percentage < 1.0:
            X_reduced, _, y_reduced, _ = train_test_split(data[modality], labels[modality], test_size=1-reduction_percentage, stratify=labels[modality])
        else:
            X_reduced, y_reduced = data[modality], labels[modality]
        
        # Perform stratified split
        X_train, X_val, y_train, y_val = train_test_split(X_reduced, y_reduced, test_size=test_and_val_size, stratify=y_reduced)
        
        train_data[modality].extend(X_train)
        train_labels[modality].extend(y_train)
        val_data[modality].extend(X_val)
        val_labels[modality].extend(y_val)
    
    return train_data, train_labels, val_data, val_labels

def plot_and_save_roc_curve(fpr_list, tpr_list, roc_auc_list, file_path):
    # Set Seaborn style
    sns.set_theme(style="darkgrid", palette="deep")
    plt.figure()

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr /= len(tpr_list)
    mean_auc = np.mean(roc_auc_list)
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label=f'Average ROC curve (area = {mean_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Save the plot as a PNG file
    plt.savefig(file_path)
    plt.close()

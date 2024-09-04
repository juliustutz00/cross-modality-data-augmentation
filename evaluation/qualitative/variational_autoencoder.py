from cross_modality_data_augmentation.transformations import CrossModalityTransformations
from cross_modality_data_augmentation.enums import Input_Modality, Output_Modality
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
from math import log10
import os



class MedicalImageDataset(Dataset):
    def __init__(self, data, labels, to_be_augmented):
        self.data = data
        self.labels = labels
        self.transform = None
        self.to_be_augmented = to_be_augmented
        self.indices = [i for i in range(len(data))]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        image = self.data[i]
        label = self.labels[i]
        
        if self.to_be_augmented:
            transform = self.choose_transformation(output_modality)
            image = transform(image)
        else:
            transform = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                ])
            image = transform(image)

        image = image.unsqueeze(0)
        return image, label

    def choose_transformation(self, modality):
        if modality=='PET':
            x = Input_Modality.PET
            if output_modality=="MRI":
                y = Output_Modality.MRI
            else:
                y = Output_Modality.CT
        elif modality=='MRI':
            x = Input_Modality.MRI
            if output_modality=="PET":
                y = Output_Modality.PET
            else:
                y = Output_Modality.CT
        elif modality=='CT':
            x = Input_Modality.CT
            if output_modality=="PET":
                y = Output_Modality.PET
            else:
                y = Output_Modality.MRI
        else:
            raise ValueError("Input modality not implemented.")

        cross_modality_transformer = CrossModalityTransformations(
        input_modality=x, 
        output_modality=y, 
        transformation_probability=1.0,
        atLeast=2, 
        atMost=3,
        color_probability=0.95, color_ratio_range=(0.75, 1), 
        artifact_probability=0.05, artifact_ratio_range=(0, 1), 
        spatial_resolution_probability=0.6, spatial_resolution_ratio_range=(0.75, 1), 
        noise_probability=0.2, noise_ratio_range=(0, 1)
        )

        return v2.Compose([
                cross_modality_transformer.transform,
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                ])

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)  # Mean vector
        self.fc22 = nn.Linear(hidden_dim, z_dim)  # Std vector
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 256*256)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x.view(-1, 1, 256, 256), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    x = x.view_as(recon_x)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch, loader, model, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

def calculate_mae(recon_batch, data):
    mae = F.l1_loss(recon_batch, data, reduction='mean').item()
    return mae

def calculate_rmse(recon_batch, data):
    mse = F.mse_loss(recon_batch, data, reduction='mean').item()
    rmse = np.sqrt(mse)
    return rmse

def test(loader, model, device):
    model.eval()
    test_loss = 0
    maes, rmses = [], []
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            recon = recon.view_as(data)
            test_loss += loss_function(recon, data, mu, logvar).item()

            # Calculate metrics
            maes.append(calculate_mae(recon, data))
            rmses.append(calculate_rmse(recon, data))

    test_loss /= len(loader.dataset)
    result_file.write("Test set loss: %f \r\n" % (test_loss))
    result_file.write("MAE: %f, RMSE: %f \r\n" % (np.mean(maes), np.mean(rmses)))

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modalities = ['CT', 'PET', 'MRI']

with open("VAE.txt", "w+", encoding="utf-8") as result_file:
    for input_modality in modalities:
        for output_modality in modalities:
            if input_modality == output_modality:
                continue
            
            result_file.write("%s to %s: \r\n" % (input_modality, output_modality))
            data_input, label_input = load_data("path/to/your/data" + input_modality)
            data_output, label_output = load_data("path/to/your/data" + output_modality)

            dataset_input = MedicalImageDataset(data_input, label_input, False)
            dataset_input_augmented = MedicalImageDataset(data_input, label_input, True)
            dataset_output = MedicalImageDataset(data_output, label_output, False)

            batch_size = 32
            loader_input = DataLoader(dataset_input, batch_size=batch_size, shuffle=True)
            loader_input_augmented = DataLoader(dataset_input_augmented, batch_size=batch_size, shuffle=True)
            loader_output = DataLoader(dataset_output, batch_size=batch_size, shuffle=True)

            vae_1 = VAE(256*256, 400, 20).to(device)
            optimizer = torch.optim.Adam(vae_1.parameters(), lr=1e-3)
            for epoch in range(1, 50):
                train(epoch, loader_input, vae_1, optimizer, device)
            result_file.write("Trained with non-augmented dataset: \r\n")
            test(loader_output, vae_1, device)

            vae_2 = VAE(256*256, 400, 20).to(device)
            optimizer = torch.optim.Adam(vae_2.parameters(), lr=1e-3)
            for epoch in range(1, 50):
                train(epoch, loader_input_augmented, vae_2, optimizer, device)
            result_file.write("Trained with augmented dataset: \r\n")
            test(loader_output, vae_2, device)
        break

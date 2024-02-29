from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pandas as pd

class CustomDataset(Dataset):

    def __init__(self, data_path, labels_path):
        # data loading
        x = pd.read_csv(data_path)
        y = pd.read_csv(labels_path)
        self.x = torch.tensor(x.to_numpy()) # tensor with the 184 values
        self.y = torch.tensor(y.to_numpy(), dtype=torch.long)  # tensor with the labels
        self.n_samples = x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import math
import sys
import os

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

class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).double()
        self.drop = nn.Dropout(0.)
        # self.relu = nn.ReLU()
        self.relu = nn.Hardswish()
        self.fc2 = nn.Linear(hidden_size, num_classes).double()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def Train(model,train_loader,criterion,optimizer):
    correct = 0
    total = 0
    for i, (values, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        values = values.to(device)
        labels = labels.double().to(device)

        labels = labels.squeeze()
        
        # Forward pass
        outputs = model(values)
        loss = criterion(outputs, labels.long())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = (100 * correct / total)

    return loss.item(), accuracy

def Validate(model,val_loader,criterion):
  with torch.no_grad():
      correct = 0
      total = 0
      for values, labels in val_loader:
          values = values.to(device)
          labels = labels.double().to(device)

          labels = labels.squeeze()

          # Forward pass
          outputs = model(values)
          loss = criterion(outputs, labels.long())
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

      accuracy = (100 * correct / total)
  return loss.item(),accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using {device}...") # gpu?

def main(mode, batch_size, learning_rate, num_epochs):

    input_size = 24
    hidden_size = 8
    num_classes = 2
    testing_size = 0.2

    transform = transforms.Compose([
     transforms.ToTensor(),
    ])

   
    # labels paths are constant and non variable
    labels_train = "/home/ivang/hayat_data_m/ready/train/y/y.csv"
    labels_val = "/home/ivang/hayat_data_m/ready/val/y/y.csv"

    # freqs paths are variable
    freqs_train = os.path.join("/home/ivang/hayat_data_m/ready/train/",f"{mode}/{mode}.csv")
    freqs_val = os.path.join("/home/ivang/hayat_data_m/ready/val/",f"{mode}/{mode}.csv")

    '''
    labels_train = "/home/ivan/Documentos/ivan/ready/train/y/y.csv"
    labels_val = "/home/ivan/Documentos/ivan/ready/val/y/y.csv"
    freqs_train = os.path.join("/home/ivan/Documentos/ivan/ready/train/",f"{mode}/{mode}.csv")
    freqs_val = os.path.join("/home/ivan/Documentos/ivan/ready/val/",f"{mode}/{mode}.csv")
    '''

    df = CustomDataset(freqs_train, labels_train)
    val_df = CustomDataset(freqs_val, labels_val)
    
    train_df, test_df = train_test_split(df, test_size = testing_size)
    
    # Data loaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(dataset=train_df, batch_size=batch_size, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(dataset=val_df, batch_size=batch_size, shuffle=False)

    # The test_loader remains unchanged for the test set
    test_loader = torch.utils.data.DataLoader(dataset=test_df, batch_size=batch_size, shuffle=False)

    model = FCNet(input_size, hidden_size, num_classes).to(device)

    # Hyper-parameters
    #learning_rate = 0.001
    # optimizer
    criterion = nn.CrossEntropyLoss()
    #Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Hyper-parameters
    #num_epochs = 100
    T_loss= []
    v_loss =[]
    T_acc = []
    v_acc = []

    # run
    for epoch in range(num_epochs):

        train_loss, train_acc = Train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = Validate(model, val_loader, criterion)

        print ('Epoch [{}/{}], train Loss: {:.4f}, train acc: {:.4f}, val Loss: {:.4f} , val acc: {:.4f}'
                    .format(epoch+1, num_epochs, train_loss,train_acc, val_loss,val_acc))
        T_loss.append(train_loss)
        v_loss.append(val_loss)
        T_acc.append(train_acc)
        v_acc.append(val_acc)

    predict = []
    with torch.no_grad():
        correct = 0
        total = 0
        for values, labels in test_loader:
            values = values.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            outputs = model(values)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predict.extend(predicted.tolist())

        print('Accuracy of the network: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    try:
        mode = sys.argv[1]
        batch_size = sys.argv[2]
        learning_rate = sys.argv[3]
        num_epochs = sys.argv[4]
    except:
        print('Please use: python3 cnn.py <mode:[x,dx,d2x]> <batch_size> <learning_rate> <n_epochs>')
    else:
        main(mode, int(batch_size), float(learning_rate), int(num_epochs))

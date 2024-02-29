import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sys
import wandb
import os
from datetime import datetime

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
        self.drop1 = nn.Dropout(0.)
        self.hard1 = nn.Hardswish()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2).double()
        self.drop2 = nn.Dropout(0.)
        self.hard2 = nn.Hardswish()
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4).double()
        self.drop3 = nn.Dropout(0.)
        self.hard3 = nn.Hardswish()
        self.fc4 = nn.Linear(hidden_size//4, num_classes).double()

    def forward(self, x):
        out = self.fc1(x)
        out = self.hard1(out)
        out = self.fc2(out)
        out = self.hard2(out)
        out = self.fc3(out)
        out = self.hard3(out)
        out = self.fc4(out)
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
print(f"Using {device}...") # gpu?

def main(mode, batch_size, learning_rate, num_epochs):

    date = datetime.now()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="moustache",
        name=f"{date.day}/{date.month}/{date.year}||{date.hour}:{date.minute}",
    
        # track hyperparameters and run metadata
        config = {
        "mode":mode,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "architecture": "xap4h",
        "epochs": num_epochs,
        }
    )

    #xap3h --> 24, 16, 8, 2
    #xap4h --> 24, 16, 8, 4, 2
    input_size = 24
    hidden_size = 16
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
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})

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
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()
    
    decision = input("Do you want to save the model? [y/n]")
    if decision == "y":
        model_name = f'{mode}_model.pth'
        torch.save(model.state_dict(), model_name)
        print("Model saved!")

if __name__ == '__main__':
    try:
        mode = sys.argv[1]
        batch_size = sys.argv[2]
        learning_rate = sys.argv[3]
        num_epochs = sys.argv[4]
    except:
        print('Please use: python3 _mcopy_wandb.py <mode:[x,dx,d2x]> <batch_size> <learning_rate> <n_epochs>')
    else:
        main(mode, int(batch_size), float(learning_rate), int(num_epochs))

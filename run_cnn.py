import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import sys
import os
import wandb
from datetime import datetime
# -----------------------------------
from models import FCNet
from data import CustomDataset
from utilities import Train, Validate, read_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}...") # check device running code

def main(data_dir, mode, batch_size, learning_rate, num_epochs):

    date = datetime.now()

    # start a new wandb run to track this script
    wandb.init(
        
        # set the wandb project where this run will be logged
        project = "moustache",
        name = f"{date.day}/{date.month}/{date.year}||{date.hour}:{date.minute}",
    
        # track hyperparameters
        config = {
        "mode": mode,
        "batch_size": batch_size,
        "lr": learning_rate,
        "arch": "xap4h",
        "epochs": num_epochs,
        }
    )

    # labels paths are constant and non variable but freqs paths will depend on what mode we run the train session
    labels_train = os.path.join(data_dir,"ready/train/y/y.csv")
    labels_val = os.path.join(data_dir,"ready/val/y/y.csv")
    freqs_train = os.path.join(data_dir,f"ready/train/{mode}/{mode}.csv")
    freqs_val = os.path.join(data_dir,f"ready/val/{mode}/{mode}.csv")

    df = CustomDataset(freqs_train, labels_train)
    val_df = CustomDataset(freqs_val, labels_val)
    
    train_df, test_df = train_test_split(df, test_size = 0.2)
    
    # Data loaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(dataset=train_df, batch_size=batch_size, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(dataset=val_df, batch_size=batch_size, shuffle=False)

    # The test_loader remains unchanged for the test set
    test_loader = torch.utils.data.DataLoader(dataset=test_df, batch_size=batch_size, shuffle=False)

    #CNN parameters
    input_size = 24
    hidden_size = 8
    num_classes = 2

    model = FCNet(input_size, hidden_size, num_classes).to(device)

    # optimizer
    criterion = nn.CrossEntropyLoss()
    #Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # run
    for epoch in range(num_epochs):

        train_loss, train_acc = Train(model, train_loader, criterion, optimizer,device)
        val_loss, val_acc = Validate(model, val_loader, criterion,device)

        print ('Epoch [{}/{}], train Loss: {:.4f}, train acc: {:.4f}, val Loss: {:.4f} , val acc: {:.4f}'
                    .format(epoch+1, num_epochs, train_loss,train_acc, val_loss,val_acc))

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
        wandb.finish()
    
    decision = input("Do you want to save the model generated in this training session? [yes/no]: ")
    if decision =="yes":
        model_name = f'{mode}_model.pth'
        torch.save(model.state_dict(), model_name)
        print(f'{model_name} saved!')

if __name__ == '__main__':
    try:
        config_d = read_config(sys.argv[1])
    except:
        print('Please use: python3 run_cnn.py <config_file.yaml>')
    else:
        print(f'Training sesion parameters --> mode:{config_d["mode"]}, batch_size:{config_d["batch_size"]}, lr:{config_d["lr"]}, epochs:{config_d["epochs"]}')
        main(config_d['data_dir'], config_d['mode'], int(config_d['batch_size']), float(config_d['lr']), int(config_d['epochs']))
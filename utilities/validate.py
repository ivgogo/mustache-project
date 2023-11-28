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
import os

def Validate(model,val_loader,criterion,device):
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
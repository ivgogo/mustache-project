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
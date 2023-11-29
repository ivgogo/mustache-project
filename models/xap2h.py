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

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).double()
        self.drop = nn.Dropout(0.)
        self.hard = nn.Hardswish()
        self.fc2 = nn.Linear(hidden_size, num_classes).double()

    def forward(self, x):
        out = self.fc1(x)
        out = self.hard(out)
        out = self.fc2(out)
        return out
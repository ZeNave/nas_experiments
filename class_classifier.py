import sys
sys.path.append("/home/josenave/Desktop/EPE-NAS")

import os
import time
import argparse
import random
import numpy as np
import math
import pandas as pd
import tabulate

from tqdm import trange
from statistics import mean
from scipy import stats
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset

from datasets import get_datasets
from config_utils import load_config
from nas_201_api import NASBench201API as API

from torch import nn


class Classifier_2_512(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(batch_size*3072, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Classifier_3_512(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(batch_size*3072, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Classifier_3_512_pca(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Classifier_6_1024(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(batch_size*3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()

        # In_channel: 3, out_channel: 16, kernel_size: 8, stride: 4
        self.conv1 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.relu1 = nn.ReLU()

        # In_channel: 16, out_channel: 32, kernel_size: 4, stride: 2
        self.conv2 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.relu2 = nn.ReLU()

        # FC
        self.fc1 = nn.Linear(1024, 512) 
        self.relu3 = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.5) 

        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f"x.shape original {x.shape}") # x.shape original torch.Size([batch_size, 256, 3, 32, 32]) 
        x = self.conv1(x)
        # print(f"x.shape after conv1 {x.shape}") # x.shape after conv1 torch.Size([batch_size, 512, 3, 16, 16])
        x = self.relu1(x)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        # print(f"x.shape after max_pool3d_1 {x.shape}") # x.shape after max_pool3d_1 torch.Size([64, 512, 1, 8, 8])

        x = self.conv2(x)
        # print(f"x.shape after conv2 {x.shape}") # x.shape after conv2 torch.Size([64, 1024, 1, 4, 4])
        x = self.relu2(x)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        # print(f"x.shape after max_pool3d_2 {x.shape}") # x.shape after max_pool3d_2 torch.Size([64, 1024, 1, 2, 2])
        x = x.view(batch_size, -1)
        # print(f"x.shape after flatten {x.shape}") # x.shape after flatten torch.Size([64, 4096])
        x = self.fc1(x) 
        # print(f"x.shape after fully_connected_1 {x.shape}") # x.shape after fully_connected_1 torch.Size([64, 1024])
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # print(f"x.shape after fully_connected_2 {x.shape}") # x.shape after fully_connected_2 torch.Size([64, 1])
        x = x.squeeze()
        # print(x)
        return x
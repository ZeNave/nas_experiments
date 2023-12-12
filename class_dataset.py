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
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import Dataset
from models import get_cell_based_tiny_net

import torchvision.transforms as transforms
from datasets import get_datasets
from config_utils import load_config
from nas_201_api import NASBench201API as API

class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index][0]
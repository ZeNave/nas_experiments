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
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import Dataset
from models import get_cell_based_tiny_net

import torchvision.transforms as transforms
from datasets import get_datasets
from config_utils import load_config
from nas_201_api import NASBench201API as API

from torch import nn

from my_code.class_dataset import MyDataset
from class_classifier import Classifier_2_512, Classifier_3_512, Classifier_6_1024, SimpleConvNet

from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description='EPE-NAS')
parser.add_argument('--data_loc', default='./datasets/cifar', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./datasets/NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results_train', type=str, help='folder to save results')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--evaluate_size', default=256, type=int)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=1, type=int)
parser.add_argument('--n_hidden_layers', default=3, type=int)
parser.add_argument('--size_of_hidden_layers', default=512, type=int)
parser.add_argument('--training_sets', default=28, type=int)
parser.add_argument('--number_of_datasets', default=16, type=int)
parser.add_argument('--convolution', default=0, type=int)
parser.add_argument('--epochs', default=200, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.convolution == 0:
    classifier_dict = {'classifier_2_512': Classifier_2_512(args.batch_size), 'classifier_3_512': Classifier_3_512(args.batch_size), 'classifier_6_1024': Classifier_6_1024(args.batch_size)}
    model = classifier_dict[f'classifier_{args.n_hidden_layers}_{args.size_of_hidden_layers}'].to(device)

else:
    classifier_dict = {'simple_conv_net': SimpleConvNet()}
    model = classifier_dict['simple_conv_net'].to(device)

if args.convolution == 0:
    model.load_state_dict(torch.load(f'epochs_{args.epochs}/model_{args.n_hidden_layers}_{args.size_of_hidden_layers}_{args.number_of_datasets}_{args.batch_size}.pth'))
else:
    model.load_state_dict(torch.load(f'epochs_{args.epochs}/model_conv_{args.batch_size}.pth'))

print(model)

def retrieve_y_true_pred(dataloader):
    model.eval()
    num_batches = len(dataloader)
    y_true = torch.tensor([]).to(device)
    y_pred = torch.tensor([]).to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).float()
            y = y.to(device).float()
            pred = model(X)
            pred = pred.to(device)
            y_true = torch.cat((y_true, y), 0)
            y_pred = torch.cat((y_pred, pred), 0)
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    # Calculate the r-squared score
    y_true = y_true.cpu()
    y_true = y_true.numpy()
    y_pred = y_pred.cpu()
    y_pred = y_pred.numpy()
    r_squared = r2_score(y_true, y_pred)

    return r_squared


dataloaders = {}

for i in range(args.number_of_datasets+1):
    print(i)
    if args.convolution == 0:
        dataset = torch.load(f'mine_dataset/batch_{args.batch_size}/train_{i}_{args.batch_size}.pt')
    else:
        dataset = torch.load(f'mine_dataset/batch_{args.batch_size}_conv/train_{i}_{args.batch_size}_conv.pt')
    if args.pca == 1:
        dataset = torch.load(f'mine_dataset/pca/batch_{args.batch_size}/train_{i}_{args.batch_size}.pt')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloaders[f'{i}_dataloader'] = dataloader

dataloaders = {}
y_true_total = torch.tensor([]).to(device)
y_pred_total = torch.tensor([]).to(device)
for i in range(args.number_of_datasets+1):
    print(i)
    if args.convolution == 0:
        dataset = torch.load(f'mine_dataset/train_{i}_{args.batch_size}.pt')
    else:
        dataset = torch.load(f'mine_dataset/batch_{args.batch_size}_conv/train_{i}_{args.batch_size}_conv.pt')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloaders[f'{i}_dataloader'] = dataloader
    y_true, y_pred = retrieve_y_true_pred(dataloaders[f'{i}_dataloader'])
    y_true_total = torch.cat((y_true_total, y_true), 0)
    y_pred_total = torch.cat((y_pred_total, y_pred), 0)
    if i == 0:
        print(f"y_true_total: {y_true_total} \ny_pred_total: {y_pred_total}")
        print(f"len(y_true_total): {len(y_true_total)} \nlen(y_pred_total): {len(y_pred_total)}")
        r_sq = calculate_metrics(y_true_total, y_pred_total)
        print(r_sq)
    



print(f"y_true_total: {y_true_total} \ny_pred_total: {y_pred_total}")
print(f"len(y_true_total): {len(y_true_total)} \nlen(y_pred_total): {len(y_pred_total)}")

r_sq = calculate_metrics(y_true_total, y_pred_total)
print(r_sq)

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
from class_classifier import Classifier_2_512, Classifier_3_512, Classifier_6_1024, SimpleConvNet, Classifier_3_512_pca
from sklearn.decomposition import PCA

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
parser.add_argument('--pca', default=0, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.pca == 1:
    classifier_dict = {'classifier_3_512': Classifier_3_512_pca(args.batch_size)}
    model = classifier_dict[f'classifier_{args.n_hidden_layers}_{args.size_of_hidden_layers}'].to(device)
elif args.convolution == 0:
    classifier_dict = {'classifier_2_512': Classifier_2_512(args.batch_size), 'classifier_3_512': Classifier_3_512(args.batch_size), 'classifier_6_1024': Classifier_6_1024(args.batch_size)}
    model = classifier_dict[f'classifier_{args.n_hidden_layers}_{args.size_of_hidden_layers}'].to(device)

else:
    classifier_dict = {'simple_conv_net': SimpleConvNet()}
    model = classifier_dict['simple_conv_net'].to(device)

print(model)

learning_rate = 1e-3    
batch_size = 64
epochs = 200

criterion = nn.MSELoss()
criterion = criterion.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # print(f"X.shape: {X.shape}")
        # print(f"X: {X[0]}")
        # print(f"y.shape: {y.shape[0]}")
        # print(f"y: {y}")
        # Compute prediction and loss
        X = X.to(device).float()
        y = y.to(device).float()
        pred = model(X)
        pred = pred.to(device)
        pred = pred.squeeze()
        loss = loss_fn(pred, y)
        loss = loss.to(device)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"Train/Validation Error: \nloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # print(f"X.shape: {X.shape}\nX[0]: {X[0]}\npred.shape: {pred.shape}\npred: {pred}")


def test_loop(dataloader, model, loss_fn, print_pred_and_y, last):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        y_true_total = torch.tensor([]).to(device)
        y_pred_total = torch.tensor([]).to(device)
        for X, y in dataloader:
            X = X.to(device).float()
            y = y.to(device).float()
            pred = model(X)
            pred = pred.to(device)
            pred = pred.squeeze()
            test_loss += loss_fn(pred, y).item()
            y_true_total = torch.cat((y_true_total, y), 0)
            y_pred_total = torch.cat((y_pred_total, pred), 0)
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    #correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    if print_pred_and_y:
        print(f"\ntrue label: {y} \npredicted: {pred}")
    elif last:
        print(f"\ntrue label: {y} \npredicted: {pred}")

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

print(f"training dataset {args.number_of_datasets}")
for t in range(epochs):
    print(f"Epoch {t+1} \n------------------------------- ")
    for n in range(1):#args.number_of_datasets):
        #print(f'training dataset {n}/{args.number_of_datasets}')
        train_loop(dataloaders[f'{n}_dataloader'], model, criterion, optimizer)
    print("testing")
    if (t+1) % 20 == 0:
        test_loop(dataloaders[f'{args.number_of_datasets}_dataloader'], model, criterion, True, False)
    elif (t+1) == 200:
        test_loop(dataloaders[f'{args.number_of_datasets}_dataloader'], model, criterion, False, True)
    else:
        test_loop(dataloaders[f'{args.number_of_datasets}_dataloader'], model, criterion, False, False)

print("Done!")
if args.pca == 1:
    torch.save(model.state_dict(), f'epochs_{epochs}/model_pca_{args.n_hidden_layers}_{args.size_of_hidden_layers}_{args.number_of_datasets}_{args.batch_size}.pth')
elif args.convolution == 1:
    torch.save(model.state_dict(), f'epochs_{epochs}/model_conv_{args.batch_size}.pth')
else:
    torch.save(model.state_dict(), f'epochs_{epochs}/model_{args.n_hidden_layers}_{args.size_of_hidden_layers}_{args.number_of_datasets}_{args.batch_size}.pth')

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

y_true_total = torch.tensor([]).to(device)
y_pred_total = torch.tensor([]).to(device)
for i in range(args.number_of_datasets+1):
    print(i)

    y_true, y_pred = retrieve_y_true_pred(dataloaders[f'{i}_dataloader'])
    y_true_total = torch.cat((y_true_total, y_true), 0)
    y_pred_total = torch.cat((y_pred_total, y_pred), 0)
    if i == 0:
        # print(f"y_true_total: {y_true_total} \ny_pred_total: {y_pred_total}")
        print(f"len(y_true_total): {len(y_true_total)} \nlen(y_pred_total): {len(y_pred_total)}")
        r_sq = calculate_metrics(y_true_total, y_pred_total)
        print(f"r_sq:   {r_sq}")
    

# print(f"y_true_total: {y_true_total} \ny_pred_total: {y_pred_total}")
print(f"len(y_true_total): {len(y_true_total)} \nlen(y_pred_total): {len(y_pred_total)}")

r_sq = calculate_metrics(y_true_total, y_pred_total)
print(f"r_sq:   {r_sq}")

def calculate_top_10s(y_pred, y_true):
    if len(y_pred.shape) == 2:
        print(y_pred.shape)
        y_pred = np.squeeze(y_pred, axis=1)

    argssorted = np.argsort(y_pred)
    top_10_ind = argssorted[-10:]
    not_top_10_ind = argssorted[:10]
    print(f"top_10_pred: {y_pred[top_10_ind]}\ntop_10_pred_true: {y_true[top_10_ind]}")
    print(f"not_top_10_pred: {y_pred[not_top_10_ind]}\nnot_top_10_pred_true: {y_true[not_top_10_ind]}\n")

    argssorted = np.argsort(y_true)
    top_10_ind = argssorted[-10:]
    not_top_10_ind = argssorted[:10]
    print(f"\ntop_10_true_pred: {y_pred[top_10_ind]}\ntop_10_true: {y_true[top_10_ind]}")
    print(f"not_top_10_true_pred: {y_pred[not_top_10_ind]}\nnot_top_10_true: {y_true[not_top_10_ind]}")




order_fn = np.nanargmax
order_fn_inv = np.nanargmin
y_pred_total = y_pred_total.cpu()
y_true_total = y_true_total.cpu()
i_max = order_fn(y_pred_total)
i_min = order_fn_inv(y_pred_total)
print("y_pred_total[i_max]: ",y_pred_total[i_max])
print("y_pred_total[i_min]: ",y_pred_total[i_min])
print("y_true_total[i_max]: ", y_true_total[i_max])
print("y_true_total[i_min]: ", y_true_total[i_min])

calculate_top_10s(y_pred_total, y_true_total)

    # for batch_data, batch_labels in dataloader:
    #     print("\nAnother batch in the dataloader")
    #     features, labels = next(iter(dataloader))
    #     print(f"\nFeature batch shape: {features.size()}")
    #     print(f"Labels batch shape: {labels.size()}")

    #     X = features.to(device)
    #     logits = model(X)
    #     logits = logits.squeeze()
    #     print(f"Predicted logits shape: {logits.size()}")
    #     print(f"Predicted logits[:10]: {logits[:10]}")
    #     print(f"True labels[:10]: {labels[:10]}\n")
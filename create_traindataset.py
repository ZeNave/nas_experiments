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
from my_code.class_dataset import MyDataset
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description='EPE-NAS')
parser.add_argument('--data_loc', default='../datasets/cifar', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../datasets/NAS-Bench-201-v1_0-e61699.pth',
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
parser.add_argument('--convolution', default=0, type=int)
parser.add_argument('--pca', default=0, type=int)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dset = args.dataset if not args.trainval else 'cifar10-valid'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(args)
api = API(args.api_loc, verbose=False)

os.makedirs(args.save_loc, exist_ok=True)

def get_batch_jacobian(net, x, target, to, device, args=None):
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()#, grad

train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0)

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'

else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

if args.trainval:
    cifar_split = load_config('../config_utils/cifar-split.txt', None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               num_workers=0, pin_memory=True, sampler= torch.utils.data.sampler.SubsetRandomSampler(train_split))

else:
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True)

indices = np.random.randint(0,15625,args.n_samples)

#indices_dict = {'train1_1': indices[:500], 'train1_2': indices[500:1000], 'train2_1': indices[1000:1500], 'train2_2': indices[500:1000], 'test': indices[1000:]}
indices_dict = {}

training_sets = 28

for i in range(training_sets):
    indices_dict[f'train_{i}'] = indices[i*250:(i+1)*250]

#print(indices_dict)


#accs      = []

def perform_pca(dataset, batch_size, key):
    if batch_size != 250:
        batch_size = args.number_of_datasets*250 + 250

    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    for X, y in dataloader:
        print(X.shape)
        print(y.shape)
        X = X.reshape(batch_size, 256 * 3072)
        # print(X.shape)
        pca = PCA(n_components=64)
        X_pca = pca.fit_transform(X)
        # print(X_pca.shape)
        X_pca = X_pca.reshape(batch_size, 8, 8)
        # print(X_pca.shape)
        y = y.unsqueeze(1)
        # print(y.shape)
        if batch_size != 250:
            for i in range(args.number_of_datasets+1):
                print(i*250,(i+1)*250)
                print(X_pca[i*250:(i+1)*250].shape, y[i*250:(i+1)*250].shape)
                dataset = MyDataset(X_pca[i*250:(i+1)*250], y[i*250:(i+1)*250])
                torch.save(dataset, f'mine_dataset/pca/batch_{args.batch_size}/train_{i}_{args.batch_size}.pt')
                dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
                dataloaders[f'{i}_dataloader'] = dataloader
        else:
            print(X_pca.shape)
            print(y.shape)
            dataset = MyDataset(X_pca, y)
            torch.save(dataset, f'mine_dataset/pca/batch_{args.batch_size}/{key}_{args.batch_size}.pt')
            
    return X_pca, y

train_x_total = torch.tensor([])
train_y_total = torch.tensor([])

for key in indices_dict.keys(): # ['train', 'validation', 'test']
    scores = []
    train_x = []
    train_y = []
    print(key)
    print(len(indices_dict[key]))
    for arch in indices_dict[key]:
        data_iterator = iter(train_loader)
        x, target = next(data_iterator)
        x, target = x.to(device), target.to(device)
        config = api.get_net_config(arch, args.dataset)
        #config['num_classes'] = 10

        network = get_cell_based_tiny_net(config)  # create the network from configuration
        network = network.to(device)

        jacobs = []
        targets = []
        grads = []
        iterations = int(np.ceil(args.batch_size/args.batch_size))
        
        for i in range(iterations):
            jacobs_batch, target = get_batch_jacobian(network, x, target, None, None)
            jacobs.append(jacobs_batch.cpu().numpy())
            info = api.query_by_index(arch)
            targets.append(info.get_metrics(dset, val_acc_type)['accuracy']*0.01)
        jacobs = np.concatenate(jacobs, axis=0)
        if(jacobs.shape[0]>args.evaluate_size):
            jacobs = jacobs[0:args.evaluate_size, :]
        
        train_x.append(jacobs)
        train_y.append(targets)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print(f"{key}_x.shape: {train_x.shape}\n")
    print(f"{key}_y.shape: {train_y.shape}\n")

    traindata = MyDataset(train_x, train_y)

    trainloader = DataLoader(traindata, batch_size=10, shuffle=True)

    train_features, train_labels = next(iter(trainloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    if args.pca == 1:
        train_x, train_y = perform_pca(traindata, 250, key)
        traindata = MyDataset(train_x, train_y)
        # train_x_total = torch.cat((train_x_total, train_x), 0).to('cpu')
        # train_y_total = torch.cat((train_y_total, train_y), 0).to('cpu')
        # print(train_x_total.shape)
        # print(train_y_total.shape)
        dataloader = DataLoader(traindata, batch_size=64, shuffle=True)
        # dataloaders[f'{i}_dataloader'] = dataloader
    elif args.convolution == 1:
        torch.save(traindata, f'mine_dataset/batch_{args.batch_size}_conv/{key}_{args.batch_size}_conv.pt')
    else:
        torch.save(traindata, f'mine_dataset/batch_{args.batch_size}/{key}_{args.batch_size}.pt')
    print('oi')

if args.pca == 1:
    # train_y_total = train_y_total.unsqueeze(dim=1)
    print(train_x_total.shape)
    print(train_y_total.shape)
    traindata = MyDataset(train_x_total, train_y_total)
    dataset = perform_pca(traindata)

# train_dataset = torch.load('my_code/mine_dataset/train.pt')

# print('oi')

# trainloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# print('oi')

# train_features, train_labels = next(iter(trainloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
dataloaders = {}

def create_train_dataset_pca():
    dataset_pca_total = torch.tensor([])
    y_pca_total = torch.tensor([])
    for i in range(args.number_of_datasets+1):
        print(i)
        dataloader = DataLoader(dataset, batch_size=250, shuffle=True)
        for X, y in dataloader:
            dataset_pca_total = torch.cat((dataset_pca_total, X), 0)
            y_pca_total = torch.cat((y_pca_total, y), 0)
            print(dataset_pca_total.shape)
            print(y_pca_total.shape)
    
    if args.pca == 1:
        y_pca_total = y_pca_total.unsqueeze(dim=1)
        traindata = MyDataset(dataset_pca_total, y_pca_total)
        dataset = perform_pca(traindata)
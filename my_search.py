import os
import time
import argparse
import random
import numpy as np
import math
import pandas as pd
import tabulate
import statistics

from tqdm import trange
from statistics import mean
from scipy import stats
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from class_classifier import Classifier_2_512, Classifier_3_512, Classifier_6_1024, SimpleConvNet, Classifier_3_512_pca



parser = argparse.ArgumentParser(description='EPE-NAS')
parser.add_argument('--data_loc', default='./datasets/cifar', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./datasets/NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
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
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--convolution', default=0, type=int)
parser.add_argument('--pca', default=0, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim

from models import get_cell_based_tiny_net

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

import torchvision.transforms as transforms
from datasets import get_datasets
from config_utils import load_config
from nas_201_api import NASBench201API as API

print(args)
# make code device agnostic but preferably make default device gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_info_best_arch(indices, order_fn_scores):

    best_arch = indices[order_fn_scores] # takes the indice of the architecture with highest estimated performance
    info = api.query_by_index(best_arch) # info: ArchResults(arch-index=12172, arch=|avg_pool_3x3~0|+|avg_pool_3x3~0|avg_pool_3x3~1|+|nor_conv_3x3~0|none~1|avg_pool_3x3~2|, 5 runs, clear=True)

    return best_arch, info

def get_sum_sorted_lists(l1, l2):

    l1_np = np.array(l1)
    l2_np = np.array(l2)

    sort_index_l1 = np.argsort(l1_np)
    sort_index_l2 = np.argsort(l2_np)

    l_sum_rank = []

    for i in range(len(l1_np)):
        sum_rank = np.where(sort_index_l1 == i)[0][0] + np.where(sort_index_l2 == i)[0][0]
        l_sum_rank.append(sum_rank)
    
    max_rank = order_fn(l_sum_rank)

    return max_rank, l_sum_rank


def get_batch_jacobian(network, x, target, to, device, args=None):
    '''
    x.shape: torch.Size([256, 3, 32, 32])
    target.shape:    torch.Size([256])
    y.shape:    torch.Size([256, 10])
    jacob.shape:   torch.Size([256, 3, 32, 32])
    '''

    # we want to make sure the gradients are zero, otherwise pytorch will sum the new ones with the already existing ones
    network.zero_grad() 

    # we want to calculate the jacobian of the loss with respect to the input and not to the weights of the network
    x.requires_grad_(True)

    # make a forward pass of the input in the network
    _, y = network(x)

    # the jacobain of neural network tells us how local perturbations the neural network input would impact the output.

    y.backward(torch.ones_like(y))

    # get the gradients in x
    jacob = x.grad.detach()

    return jacob, target.detach()#, grad

def eval_score_perclass_original(jacob, labels=None, n_classes=10):
    k = 1e-5
    #n_classes = len(np.unique(labels))
    per_class={}
    for i, label in enumerate(labels[0]):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            per_class[label] = jacob[i]

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        s = 0
        try:
            corrs = np.corrcoef(per_class[c])
            s = np.sum(np.log(abs(corrs)+k))#/len(corrs)
                       
            if n_classes > 100:
                s /= len(corrs)
        except: # defensive programming
            continue

        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:

        for c in ind_corr_matrix_score_keys:
            # B)
            score += np.absolute(ind_corr_matrix_score[c])
    else: 
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score += np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

        # should divide by number of classes seen
        score /= len(ind_corr_matrix_score_keys)

    return score

def eval_score_perclass_kl(jacob, labels=None, n_classes=10):

    k = 1e-5
    #n_classes = len(np.unique(labels))
    per_class={}
    for i, label in enumerate(labels[0]):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            per_class[label] = jacob[i]
    
    # per class is a dictionary. the keys are the different targets, and for each key there is a list of the jacbobians of the inputs 

    ind_corr_matrix_kl_score = {}
    for c in per_class.keys():
        s = 0
        try:
            # Return Pearson product-moment correlation coefficients.
            corrs = np.corrcoef(per_class[c]) # matrix of shape number_of_inputs_of_class_c x number_of_inputs_of_class_c

            s = np.sum(np.log(abs(corrs)+k))#/len(corrs)

            # NAS-WOT method for calculating dissimilarity between coorelation matrix 
            v, _  = np.linalg.eig(corrs)
            k = 1e-5
            s1 = -np.sum(np.log(v + k) + 1./(v + k))
            if n_classes > 100:
                s /= len(corrs)
                s1 /= len(corrs)
        except: # defensive programming
            continue

        ind_corr_matrix_kl_score[c] = s1

    # ind_corr_matrix_score is a dictionary with the correlation score for each class

    # per class-corr matrix A and B
    score1 = 0
    ind_corr_matrix_kl_score_keys = ind_corr_matrix_kl_score.keys()
    if n_classes <= 100:

        for c in ind_corr_matrix_kl_score_keys:
            # B)
            score1 += np.absolute(ind_corr_matrix_kl_score[c])
    else: 
        for c in ind_corr_matrix_kl_score_keys:
            # A)
            for cj in ind_corr_matrix_kl_score_keys:
                score1 += np.absolute(ind_corr_matrix_kl_score[c]-ind_corr_matrix_kl_score[cj])

        # should divide by number of classes seen
        score1 /= len(ind_corr_matrix_kl_score_keys)

    return score1

# def test_loop(dataloader, model, loss_fn, print_pred_and_y):
#     # Set the model to evaluation mode - important for batch normalization and dropout layers
#     # Unnecessary in this situation but added for best practices
#     model.eval()
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0

#     # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
#     # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
#     with torch.no_grad():
#         for X, y in dataloader:
#             X = X.to(device).float()
#             y = y.to(device).float()
#             pred = model(X)
#             pred = pred.to(device)
#             pred = pred.squeeze()
#             test_loss += loss_fn(pred, y).item()
#             #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#     test_loss /= num_batches
#     #correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
#     if print_pred_and_y:
#         print(f"\ntrue label: {y} \npredicted: {pred}")

def load_model(n_hidden_layers=2, size_of_hidden_layers=512, training_sets=8, number_of_datasets=16, epochs=200):
    if args.pca == 1:
        classifier_dict = {'classifier_pca_3_512': Classifier_3_512_pca(args.batch_size)}
        model = classifier_dict[f'classifier_pca_{args.n_hidden_layers}_{args.size_of_hidden_layers}'].to(device)
    elif args.convolution == 0:
        classifier_dict = {'classifier_2_512': Classifier_2_512(args.batch_size), 'classifier_3_512': Classifier_3_512(args.batch_size), 'classifier_6_1024': Classifier_6_1024(args.batch_size)}
        model = classifier_dict[f'classifier_{args.n_hidden_layers}_{args.size_of_hidden_layers}'].to(device)
    else:
        classifier_dict = {'simple_conv_net': SimpleConvNet()}
        model = classifier_dict['simple_conv_net'].to(device)

    #model = classifier_dict[f'classifier_{n_hidden_layers}_{size_of_hidden_layers}'].to(device)
    if args.pca == 1:
        model.load_state_dict(torch.load(f'epochs_{args.epochs}/model_pca_{args.n_hidden_layers}_{args.size_of_hidden_layers}_{args.number_of_datasets}_{args.batch_size}.pth'))
    elif args.convolution == 0:
        model.load_state_dict(torch.load(f'epochs_{args.epochs}/model_{args.n_hidden_layers}_{args.size_of_hidden_layers}_{args.number_of_datasets}_{args.batch_size}.pth'))
    else:
        model.load_state_dict(torch.load(f'epochs_{args.epochs}/model_conv_{args.batch_size}.pth'))
    return model

def model_score(model, jacob, labels=None, n_classes=10):

    score = model(jacob)
    score = score.item()

    return score

# NAS-WOT method, for comparison
def eval_score(jacob, labels=None, n_classes=10):

    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    score = -np.sum(np.log(v + k) + 1./(v + k))
    return score

# start time to now how long the program took to run
THE_START = time.time()


api = API(args.api_loc, verbose=False) # api_loc='./datasets/NAS-Bench-201-v1_1-096897.pth'

os.makedirs(args.save_loc, exist_ok=True) # save_loc='results' ; create directory to save results

train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0) # dataset='cifar10' ; data_loc='./datasets/cifar10' ; returns dataset with pre-defined transform already applied


# acc_type ? val_acc_type?
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
    
val_acc   = []
times     = []
chosen = []
acc = []
best_val_acc = []
topscores = []
model_pred_acc_of_best_arch = []

dset = args.dataset if not args.trainval else 'cifar10-valid' # args.dataset cifar10

order_fn = np.nanargmax
order_fn_inv = np.nanargmin

runs = trange(args.n_runs, desc='') # n_runs=1

# s1s = []
# s2s = []
# The following values were obtained for 3000 random samples 
s1_mean = 22694.075841465932
s1_std_dev = 6560.258266839199

s2_mean = -11564.55637175022
s2_std_dev = 60788.12793359157

alpha = 0.1

model = load_model(n_hidden_layers=args.n_hidden_layers, size_of_hidden_layers=args.size_of_hidden_layers, training_sets=args.training_sets, number_of_datasets=args.number_of_datasets, epochs=args.epochs)
print(model)

for N in runs:
    
    start = time.time() # start time  
    indices = np.random.randint(0,15625,args.n_samples) # n_samples randomly from all the architectures
    #scores_epenas = []
    #scores_naswot = []
    scores = []
    val_scores = []

    for arch in indices:
        data_iterator = iter(train_loader) 
        x, target = next(data_iterator) 
        x, target = x.to(device), target.to(device) # batch of inputs and targets      


        # takes in an idex of an architecture and the name of a dataset
        # EXAMPLE --> arch:   10820; args.datset:    cifar10
        config = api.get_net_config(arch, args.dataset) 
        # config: {'name': 'infer.tiny', 'C': 16, 'N': 5, 'arch_str': '|skip_connect~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|skip_connect~0|nor_conv_1x1~1|skip_connect~2|', 'num_classes': 10}

        network = get_cell_based_tiny_net(config)  # create the network from configuration
        network = network.to(device) #   TinyNetwork(C=16, N=5, L=17)

        jacobs = []
        targets = []
        grads = []
        iterations = int(np.ceil(args.batch_size/args.batch_size))

        for i in range(iterations):
            jacobs_batch, target = get_batch_jacobian(network, x, target, None, None)
            if args.convolution == 0:
                jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
            else:
                jacobs.append(jacobs_batch.cpu().numpy())
            # len(jacobs): 1
            # jacobs[0].shape:  (256, 3072)

            targets.append(target.cpu().numpy())
        jacobs = np.concatenate(jacobs, axis=0)
        # len(jacobs): 256
        # jacobs[0].shape:  (3072, )
        if(jacobs.shape[0]>args.evaluate_size):
            jacobs = jacobs[0:args.evaluate_size, :]
        
        tensor = torch.Tensor(jacobs)
        # print(f"tensor.shape: {tensor.shape}")
        single_input = tensor.unsqueeze(0)
        # print(f"single_input.shape: {single_input.shape}")
        jacobs = single_input.to(device)

        try:
            #s1 = eval_score_perclass_original(jacobs, targets)
            #s2 = eval_score(jacobs)
            s = model_score(model, jacobs)

        except Exception as e:
            print(e)
            s1 = np.nan
            s2 = np.nan
            s = np.nan

        # scores_kl_per_class.append(s3)
        # scores_epenas.append(s1)
        # scores_naswot.append(s2)
        scores.append(s)
        info = api.query_by_index(arch)
        val_scores.append(info.get_metrics(dset, val_acc_type)['accuracy'])
        # print(f"predicted accuracy: {s}")
        # print(f"training accuracy: {info.get_metrics(dset, acc_type)['accuracy']}")
        # print(f"validation accuracy: {info.get_metrics(dset, val_acc_type)['accuracy']}")


    # order_fn_scores, l_aplpha_sum_rank = get_sum_sorted_lists(scores_epenas, scores_naswot)
    # topscores.append(l_aplpha_sum_rank[order_fn_scores])                            # topscores:  [28857.17602543697, 28147.2694993541, 8960.95508397699]
    # print(f"scores_epenas[order_fn_scores]: {scores_epenas[order_fn_scores]}\n scores_naswot[order_fn_scores]: {scores_naswot[[order_fn_scores]]}")
    # print(f"\norder_fn_scores:  {order_fn_scores}\nl_aplpha_sum_rank[:10]:  {l_aplpha_sum_rank[:10]}\nbest_arch:  {best_arch}\n")
    order_fn_scores = order_fn(val_scores)
    best_arch, info = get_info_best_arch(indices, order_fn_scores)
    best_val_acc.append(info.get_metrics(dset, acc_type)['accuracy'])
    pred_score_best_val_acc =  scores[order_fn_scores]
    model_pred_acc_of_best_arch.append(pred_score_best_val_acc)

    order_fn_scores = order_fn(scores)
    best_arch, info = get_info_best_arch(indices, order_fn_scores)
    topscores.append(scores[order_fn_scores])
    chosen.append(best_arch)                                                    # chosen: [13349, 235, 12172]
    acc.append(info.get_metrics(dset, acc_type)['accuracy'])                    # info.get_metrics(dset, acc_type)['accuracy']: 79.315

    # print(f"indices:    {indices[:10]};\nscores_alpha:     {l_aplpha_sum_rank[:10]}\norder_fn(scores_alpha): {order_fn_scores};\nindices[order_fn(scores_alpha)]:   {indices[order_fn_scores]}")
    # print(f"\nscores_epenas:     {scores_epenas[:10]}\norder_fn(scores_epenas): {order_fn(scores_epenas)};\nindices[order_fn(scores_epenas)]:   {indices[order_fn(scores_epenas)]}")
    print(f"\nscores:     {scores[:10]}\norder_fn(scores_naswot): {order_fn_scores};\nindices[order_fn(scores_naswot)]:   {indices[order_fn_scores]}")

    if not args.dataset == 'cifar10' or args.trainval:
        val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

    times.append(time.time()-start)
    runs.set_description(f"acc: {mean(acc if not args.trainval else val_acc):.2f}%")
    print("----------------/////////////////////////-----------------")

print(f"times:  {times}")
print(f'mean time: {np.mean(times)}\n')

print(f"topscores: {topscores}")
print(f"acc:   {acc}")
print(f"val_acc:    {val_acc}\n")
# print(f"acc_kl_per_class:    {acc_kl_per_class}")
# print(f"accepenas:   {acc_epenas}")
# print(f"acc_naswot:    {acc_naswot}")

print(f"model predictions for the atual best architectures: {model_pred_acc_of_best_arch}")
print(f"best_val_acc: {best_val_acc}\n")
print(f"Final mean test accuracy: {np.mean(acc)}")
if len(val_acc) > 1:
    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

state = {'accs': acc,
         'val_accs': val_acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

dset = args.dataset if not args.trainval else 'cifar10-valid'
fname = f"{args.save_loc}/{dset}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
torch.save(state, fname)

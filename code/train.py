from __future__ import division
from __future__ import print_function

import os
import glob
import time
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from EGAT_models import SpGAT

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # List in tensor data format

    def forward(self, preds, labels):
        """
        preds: logits output values
        labels: labels
        """
        preds = F.softmax(preds, dim=1).to(device)
        eps = 1e-7
        target = self.one_hot(preds.size(1), labels).to(device)
        ce = (-1 * torch.log(preds + eps) * target).to(device)
        floss = (torch.pow((1 - preds), self.gamma) * ce).to(device)
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=16, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=6, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=20, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

data_num = 20
data_features = []
data_labels = []
data_solution = []
data_edge_features = []
data_edge_A = []
data_edge_num_A = []
data_edge_B = []
data_edge_num_B = []
data_idx_train = []
for now_data in range(data_num):
    if not os.path.exists('./data/pair' + str(now_data) + '.pickle'):
        print("No problem file!")

    with open('./data/pair' + str(now_data) + '.pickle', "rb") as f:
        problem = pickle.load(f)

    variable_features = problem[0]
    constraint_features = problem[1]
    edge_indices = problem[2]
    edge_feature = problem[3]
    optimal_solution = problem[4]
    #print(optimal_solution)
    #edge, features, labels, idx_train = load_data()

    #change
    n = len(variable_features)
    var_size = len(variable_features[0])
    m = len(constraint_features)
    con_size = len(constraint_features[0])

    edge_num = len(edge_indices[0])
    data_edge_num_A.append(edge_num)
    edge_num = len(edge_indices[0])
    data_edge_num_B.append(edge_num)

    edgeA = []
    edgeB = []
    edge_features = []
    for i in range(edge_num):
        edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
        edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])
        edge_features.append(edge_feature[i])
    edgeA = torch.as_tensor(edgeA)
    data_edge_A.append(edgeA)

    edgeB = torch.as_tensor(edgeB)
    data_edge_B.append(edgeB)

    edge_features = torch.as_tensor(edge_features)
    data_edge_features.append(edge_features)

    for i in range(m):
        for j in range(var_size - con_size):
            constraint_features[i].append(0)
    features = variable_features + constraint_features
    features = torch.as_tensor(features).float()
    data_features.append(features)

    #labelA = torch.tensor(patition_color)
    new_optimal_solution = []
    for item in optimal_solution:
        new_optimal_solution.append((int)(item))
    optimal_solution = new_optimal_solution
    num_label = [1, 1]
    num_label = torch.as_tensor(num_label).to(device)
    data_labels.append(num_label)

    for i in range(m):
        optimal_solution.append(0)

    labels = []
    #For Binary
    for i in range(n + m):
        if(optimal_solution[i] == 0):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    labels = torch.tensor(labels)

    labels = torch.as_tensor(optimal_solution)

    data_solution.append(labels)

    idx_train = torch.as_tensor(range(n))
    data_idx_train.append(idx_train)

# Model and optimizer
model = SpGAT(nfeat=data_features[0].shape[1],    # Feature dimension
            nhid=args.hidden,             # Feature dimension of each hidden layer
            nclass=int(data_solution[0].max()) + 1, # Number of classes
            dropout=args.dropout,         # Dropout
            nheads=args.nb_heads,         # Number of heads
            alpha=args.alpha)             # LeakyReLU alpha coefficient

optimizer = optim.Adam(model.parameters(),    
                       lr=args.lr,                        # Learning rate
                       weight_decay=args.weight_decay)    # Weight decay to prevent overfitting

if args.cuda: # Move to GPU
    model.to(device)
    for now_data in range(data_num):
        data_features[now_data] = data_features[now_data].to(device)
        data_labels[now_data] = data_labels[now_data].to(device)
        data_solution[now_data] = data_solution[now_data].to(device)
        data_edge_A[now_data] = data_edge_A[now_data].to(device)
        data_edge_B[now_data] = data_edge_B[now_data].to(device)
        data_edge_features[now_data] = data_edge_features[now_data].to(device)
        data_idx_train[now_data] = data_idx_train[now_data].to(device)


for now_data in range(data_num):
    data_features[now_data] = Variable(data_features[now_data])
    data_edge_A[now_data] = Variable(data_edge_A[now_data])
    data_edge_B[now_data] = Variable(data_edge_B[now_data])
    data_solution[now_data] = Variable(data_solution[now_data])
    # Define computation graph for automatic differentiation

def train(epoch, num):
    global data_edge_features
    t = time.time()

    output, data_edge_features[num] = model(data_features[num], data_edge_A[num], data_edge_B[num], data_edge_features[num].detach())
    print(data_solution[num][idx_train])

    lf = Focal_Loss(torch.as_tensor(data_labels[num]))
    loss_train = lf(output[idx_train], data_solution[num][idx_train])

    return loss_train

t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    now_loss = 0
    for i in range(5):
        now_data = random.randint(0, data_num - 1)
        now_loss += train(epoch, now_data)
    loss_values.append(now_loss)
    now_loss.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(now_loss))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:  # Stop if there's no improvement for several consecutive rounds
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

print(loss_values)


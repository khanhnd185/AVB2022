from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import torch 
import torch.nn as nn
import torch.nn.functional as F
from math import cos, pi
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import recall_score

EPS = 1e-8

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        n = float(n)
        self.sum += val * n
        self.count += n
    
    def avg(self):
        return (self.sum / self.count)

def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):

    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model

def AvgCCC(x, y):
    N = x.shape[0]
    x_m = np.mean(x, axis=0)
    y_m = np.mean(y, axis=0)
    vx = x - x_m
    vy = y - y_m
    sxy = np.sum(vx * vy, axis=0)
    sxx = np.sum(vx * vx, axis=0)
    syy = np.sum(vy * vy, axis=0)
    ccc =  2*sxy / (sxx + syy + N * (x_m - y_m)**2)
    return np.mean(ccc)

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()
    def forward(self, x, y):
        N = x.shape[0]
        x_m = torch.mean(x, dim=0)
        y_m = torch.mean(y, dim=0)
        vx = x - x_m
        vy = y - y_m
        sxy = torch.sum(vx * vy, dim=0)
        sxx = torch.sum(vx * vx, dim=0)
        syy = torch.sum(vy * vy, dim=0)
        loss = 1 - 2*sxy / (sxx + syy + N * torch.pow(x_m - y_m, 2) + EPS)
        return torch.sum(loss)

def pad_collate(batch):
    data, labels = zip(*batch)
    labels = torch.Tensor(np.array(labels))
    padded_data = pad_sequence(data, batch_first=True)
    num_seqs_per_sample = torch.Tensor([len(x) for x in data])
    
    return padded_data, labels, num_seqs_per_sample


def UAR(labels, predictions):
    predictions = np.argmax(predictions, axis = 1)
    labels = np.argmax(labels, axis = 1)
    
    predictions = predictions.astype(np.int32).reshape(-1,).tolist()
    labels = labels.astype(np.int32).reshape(-1,).tolist()
    
    return recall_score(labels, predictions, average="macro")
    
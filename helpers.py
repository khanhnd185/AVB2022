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

class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__() 
        self.loss = NegativeCCCLoss()

    def forward(self, x, y):
        loss1 = self.loss(x[:, 0], y[:, 0]) + self.loss(x[:, 1], y[:, 1])
        return loss1

class MaskedCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(MaskedCELoss, self).__init__() 
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight, ignore_index=ignore_index)
    
    def forward(self, x, y, mask):
        loss = self.ce(x, y)
        loss = loss.mean(dim=-1)
        loss = loss * mask
        return loss.sum() / (mask.sum() + EPS)

class DistillationLoss(nn.Module):
    def __init__(self, alpha, func):
        super(DistillationLoss, self).__init__() 
        self.alpha = alpha
        self.func = func
    
    def forward(self, yhat, ysoft, y, mask):
        return (1 - self.alpha) * self.func(yhat, ysoft, mask) + self.alpha * self.func(yhat, y, mask)

class NegativeCCCLoss(nn.Module):
    def __init__(self):
        super(NegativeCCCLoss, self).__init__()
    def forward(self, x, y):
        y = y.view(-1)
        x = x.view(-1)
        N = x.shape[0]
        x_m = torch.sum(x) / N
        y_m = torch.sum(y) / N
        vx = x - x_m
        vy = y - y_m
        ccc = 2*torch.dot(vx, vy) / (torch.dot(vx, vx) + torch.dot(vy, vy) + N * torch.pow(x_m - y_m, 2) + EPS)
        return 1 - ccc

def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def VA_metric(y, yhat):
    cccs = (float(CCC_score(y[:,0], yhat[:,0])), float(CCC_score(y[:,1], yhat[:,1])))
    avg_ccc = float(CCC_score(y[:,0], yhat[:,0]) + CCC_score(y[:,1], yhat[:,1])) / 2
    return avg_ccc, cccs


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
    """ Pad batch tensors to have equal length.
    
    Args:
        batch (list): Data to pad.
    
    Returns:
        padded_data (torch.Tensor): Batched data tensors. 
        labels (torch.Tensor): Batched label tensors.
        num_seqs_per_sample (list): Number of sequences of each batch tensor.
    """
    
    data, labels = zip(*batch)
    labels = torch.Tensor(np.array(labels))
    padded_data = pad_sequence(data, batch_first=True)
    num_seqs_per_sample = torch.Tensor([len(x) for x in data])
    
    return padded_data, labels, num_seqs_per_sample


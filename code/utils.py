import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import random

def str2bool(v):
    if v.lower() in ['true',1]:
        return True
    elif v.lower() in ['false',0]:
        return False
    else:
        return argparse.ArgumentTypeError('Boolean value expected.')

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count =0
    
    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mask_onehot(label,num_classes):
    mask_one_hot = []
    for i in range(num_classes): 
        temp_prob = (label == (i * torch.ones_like(label)))
        mask_one_hot.append(temp_prob)
    mask_one_hot = torch.cat(mask_one_hot, dim=1)
    return mask_one_hot.float()


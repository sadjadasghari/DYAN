import os
import sys
import torch
import torch.nn as nn
import numpy as np

def loss_berhu(x, gpu_id):
    loss = torch.empty(x.shape).cuda(gpu_id)
    c = 0.2*torch.max(x)
    loss = torch.where(torch.abs(x) <= c, torch.abs(x), (x ** 2 + c ** 2) / (2 * c))

    print(loss)
    return torch.sum(loss), c

x = torch.arange(5*5).view(5,5) - 12
loss, c = loss_berhu(x, 0)


print(loss)


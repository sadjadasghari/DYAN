import os
import sys
import torch
import numpy as np

x = torch.arange(5*5).view(5,5)
print(np.linspace(5, 1, num=5))
x = x[:, np.linspace(5, 1, num=5)-1]
print (x, type(x))
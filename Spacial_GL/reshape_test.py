import numpy as np
import torch
n = 2
i = 5

t = torch.arange(n*n).view(n, n)
tfl = t.view(n*n,-1)
tfl_exp = tfl.repeat(1,i)
t_exp = tfl_exp.view(-1,i*n).repeat(1,i).view(-1,i*n)

t_exp = t_exp[:12,:] # :15 - 3

print(t_exp)



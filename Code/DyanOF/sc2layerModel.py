############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from utils import creatRealDictionary

############################# Import Section #################################

gpu_id = 1


def fista(D, Y, Gamma,maxIter):

    DtD = torch.matmul(torch.t(D),D)
    # wen: only 2norm be the same as max(eig(DtD))! checked with Matlab
    L = torch.norm(DtD,p=2)
    if L == 0:
        print("Got L == 0")
        linv = 0
    else:
        linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    #scipy.io.savemat('DtD.mat', mdict={'DtD': DtD})
    x_old = Variable(torch.zeros(DtD.shape[1],DtY.shape[2]).cuda(gpu_id), requires_grad=True)
    t = 1
    y_old = x_old
    Gamma = Gamma*linv
    A = Variable(torch.eye(DtD.shape[1]).cuda(gpu_id),requires_grad=True) - torch.mul(DtD,linv)

    DtY = torch.mul(DtY,linv)

    lambd = Gamma.view(-1,1).expand(-1,DtY.shape[2]) #
    for ii in range(maxIter):
        Ay = torch.matmul(A,y_old)
        del y_old
        x_new = Ay +DtY - lambd
        x_new = torch.sign(x_new)*torch.max(torch.abs(x_new),torch.zeros_like(x_new))
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt)
        t = t_new
        x_old = x_new
        del x_new

    return x_old



class SClayer(nn.Module):
    def __init__(self, Drr, Dtheta, Gamma, T):
        super(SClayer, self).__init__()

        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.gamma = nn.Parameter(Gamma)
        self.T = T

    def forward(self, x):
        dic = creatRealDictionary(self.T,self.rr,self.theta)
        sparsecode = fista(dic,x,0.05,40)
        return Variable(sparsecode)

class invSClayer(nn.Module):
    def __init__(self,rr,theta, T, PRE):
        super(invSClayer,self).__init__()

        self.rr = rr
        self.theta = theta
        self.T = T
        self.PRE = PRE

    def forward(self,x):
        dic = creatRealDictionary(self.T+self.PRE,self.rr,self.theta)
        result = torch.matmul(dic,x)
        return result


class SC2(nn.Module):
    def __init__(self, Drr, Dtheta, Gamma, T, PRE):
        super(SC2, self).__init__()
        self.l1 = SClayer(Drr, Dtheta, Gamma, T)
        self.l2 = invSClayer(self.l1.rr,self.l1.theta, T, PRE)

    def forward(self,x):
        return self.l2(self.l1(x))

    def forward2(self,x):
        return self.l1(x)


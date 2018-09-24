############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

from math import sqrt
import numpy as np

############################# Import Section #################################

gpu_id = 0



# Create Dictionary
def creatRealDictionary(T,Drr,Dtheta):
	n = Drr.shape[0]
	WVar = []
	Wones = torch.ones(1).cuda(gpu_id)
	Wones  = Variable(Wones,requires_grad=False)
	for i in range(0,T):
		W1 = torch.mul(torch.pow(Drr,i) , torch.cos(i * Dtheta))
		W2 = torch.mul ( torch.pow(-Drr,i) , torch.cos(i * Dtheta) )
		W3 = torch.mul ( torch.pow(Drr,i) , torch.sin(i *Dtheta) )
		W4 = torch.mul ( torch.pow(-Drr,i) , torch.sin(i*Dtheta) )
		W = torch.cat((Wones,W1,W2,W3,W4),0)

		WVar.append(W.view(1,-1))
	dic = torch.cat((WVar),0)
	G = torch.norm(dic,p=2,dim=0)
	idx = (G == 0).nonzero()
	nG = G.clone()
	nG[idx] = np.sqrt(T) 
	G = nG
	
	dic = dic/G

	return dic

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
		x_new = Ay +DtY
		x_new = torch.sign(x_new)*torch.max(torch.abs(x_new)-lambd,torch.zeros_like(x_new))
		t0 = t
		t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
		y_old = x_new + ((t0 - 1.) / t) * (x_new - x_old)
		x_old = x_new
		del x_new

	return x_old



class Encoder(nn.Module):
	def __init__(self, Drr, Dtheta, Gamma, T):
		super(Encoder, self).__init__()
		
		self.rr = nn.Parameter(Drr)
		self.theta = nn.Parameter(Dtheta)
		self.gamma = nn.Parameter(Gamma)
		self.T = T

	def forward(self, x):
		dic = creatRealDictionary(self.T,self.rr,self.theta)
		sparsecode = fista(dic,x,0.01,40)
		return Variable(sparsecode)

class Decoder(nn.Module):
	def __init__(self,rr,theta, T, PRE):
		super(Decoder,self).__init__()

		self.rr = rr
		self.theta = theta
		self.T = T
		self.PRE = PRE

	def forward(self,x):
		dic = creatRealDictionary(self.T+self.PRE,self.rr,self.theta)
		result = torch.matmul(dic,x)
		return result


class OFModel(nn.Module):
	def __init__(self, Drr, Dtheta, Gamma, T, PRE):
		super(OFModel, self).__init__()
		self.l1 = Encoder(Drr, Dtheta, Gamma, T)
		self.l2 = Decoder(self.l1.rr,self.l1.theta, T, PRE)

	def forward(self,x):
		return self.l2(self.l1(x))

	def forward2(self,x):
		return self.l1(x)


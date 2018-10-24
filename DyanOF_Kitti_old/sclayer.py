## Author - Wen

############################# Import Section #################################

## Imports related to PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

## Generic imports
import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import hickle as hkl
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from skimage import io, transform
from matplotlib.backends.backend_pdf import PdfPages

## Dependencies classes and functions
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import getIndex
from utils import videoDataset
from utils import save_checkpoint
from utils import getListOfFolders
from utils import generateGridPoles
from utils import creatRealDictionary


## Import Model
from sc2layerModel import SC2
from sc2layerModel import SClayer

############################# Import Section #################################


checkptname = "NormDict"
## HyperParameters for the Network
num_of_poles = 40
EPOCH = 200
BATCH_SIZE = 1
LR = 0.00001				# Learning rate
DOWNLOAD_MNIST = False
print_every = 3
FRA = 9					# Training sequnece length
PRE = 1						# Prediction sequence length
N_FRAME = FRA+PRE
N = num_of_poles*4 			# Number of poles in dic
K = 5 						# PCA output atom size
T = FRA 					# Hankel length


## Initializing r, theta and Gamma
P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

Gamma = getWeights(Pall,T)
Gamma = torch.from_numpy(Gamma).float()


rootDir = '/home/abhishek/Workspace/DyanOF/Kitti_Flows/'
listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]

# Function for the PyTorch Dataloader
trainingData = videoDataset(listOfFolders=listOfFolders, 
                            rootDir=rootDir)


dataloader = DataLoader(trainingData, 
                        batch_size=BATCH_SIZE ,
                        shuffle=True, num_workers=1)


## Create the model
model = SC2(Drr, Dtheta, Gamma, T, PRE)
model.cuda(1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()
start_epoch = 1
ckpt_file = 'NormDict34.pth'
load_ckpt = True
## If want to continue training from a checkpoint
if(load_ckpt):
    loadedcheckpoint = torch.load(ckpt_file)
    start_epoch = loadedcheckpoint['epoch']
    model.load_state_dict(loadedcheckpoint['state_dict'])
    # optimizer.load_state_dict(loadedcheckpoint['optimizer'])


print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()


## Start the Training

for epoch in range(start_epoch, EPOCH+1):
    loss_value = []
    exp_lr_scheduler.step()
    for i_batch, sample in enumerate(dataloader):
        dataBatch = sample['frames'].squeeze(0)
        numBatches = dataBatch.shape[0]
        los_val = []
        for batchnum in range(numBatches):
            data = dataBatch[batchnum].cuda(1)
            expectedOut = data
            inputData = data[:,0:9,:]
            # inputData = inputData
            # print(inputData.shape)
            optimizer.zero_grad()
            output = model.forward(inputData)
            loss = loss_mse(output, expectedOut)
            loss.backward()
            optimizer.step()
            los_val.append(loss.data[0])
        loss_value.append(np.mean(np.array(los_val)))
    loss_val = np.mean(np.array(loss_value))

    if epoch % print_every ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')

    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val)

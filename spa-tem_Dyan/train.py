############################# Import Section #################################
import sys
## Imports related to PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import scipy.misc

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

## Generic imports
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

## Dependencies classes and functions
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import videoDataset
from utils import save_checkpoint
from utils import getListOfFolders

## Import Model
from DyanOF import OFModel
from DyanOF import Encoder
from DyanOF import Decoder

############################# Import Section #################################



## HyperParameters for the Network
NumOfPoles = 40
CLen = NumOfPoles*4 +1
EPOCH = 150
BATCH_SIZE = 1
LR = 0.0015
gpu_id = 1

FRA = 9 # input number of frame
PRE = 1 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA
W = 10 # Length of spacial window
saveEvery = 30

#mnist
x_fra = 128
y_fra = 160


## Load saved model 
load_ckpt = False
ckpt_file = '' # for Kitti Dataset: 'KittiModel.pth'
# checkptname = "UCFModel"
checkptname = "Kitti_Normal_lam0.1_"



## Load input data

rootDir = '/home/armandcomas/DYAN/Code/datasets/Kitti_Flows/'

listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]

trainingData = videoDataset(folderList=listOfFolders,
                            rootDir=rootDir)

dataloader = DataLoader(trainingData,
                        batch_size=BATCH_SIZE ,
                        shuffle=True, num_workers=1)

## Initializing r, theta
P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()
# What and where is gamma

## Create the time model
model_ti = OFModel(Drr, Dtheta, T, PRE, gpu_id)
model_ti.cuda(gpu_id)
optimizer = torch.optim.Adam(model_ti.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1) # if Kitti: milestones=[100,150]
loss_mse = nn.MSELoss()
start_epoch = 1

## Create the spatial model
Encoder_sp = Encoder(Drr, Dtheta, W, gpu_id)
Encoder_sp.cuda(gpu_id) #parallelize it?
optimizer_sp = torch.optim.Adam(Encoder_sp.parameters(), lr=LR)
scheduler_sp = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1) # Parameters?
loss_mse_sp = nn.MSELoss()

## If want to continue training from a checkpoint
if(load_ckpt):
    loadedcheckpoint = torch.load(ckpt_file)
    start_epoch = loadedcheckpoint['epoch']
    model.load_state_dict(loadedcheckpoint['state_dict'])
    optimizer.load_state_dict(loadedcheckpoint['optimizer'])


print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()

count = 0
# revX_idx = np.linspace(128, 1, num=128)-1
# revY_idx = np.linspace(160, 1, num=160)-1

## Start the Training
for epoch in range(start_epoch, EPOCH+1):
    t0_epoch = time.time()

    loss_value_T = []
    scheduler.step()

    for i_batch, sample in enumerate(dataloader):

        data = sample['frames'].squeeze(0).cuda(gpu_id)
        expectedOut = Variable(data)
        inputData = Variable(data[:,0:FRA,:])

        # optimizer.zero_grad()
        # output = model.forward(inputData)
        #
        # loss = loss_mse(output[:,FRA], expectedOut[:,FRA]) # if Kitti: loss = loss_mse(output, expectedOut)
        # loss.backward()
        # optimizer.step()
        # loss_value_T.append(loss.data.item())


        ##__________Spatial__________
        # to concatenate layers (encoders) is it enough propagating the loss backwards after concatenating in time?
        # is it needed to modify the model?
        C_t = []
        for i in range(FRA):
            # along columns
            inputFrame = Variable(data[:,i,:].reshape(-1,x_fra,y_fra))


            C = torch.Tensor(4, 2, CLen * y_fra * x_fra).cuda(gpu_id)

            # if we apply sliding window, c's corresponding to first window will be replicated along it
            c_list_H = [] # list of states Horizontal-Right
            c_list_HI = [] # list of states Horizontal-Left
            c_list_V = [] # list of states Vertical-Up
            c_list_VI = [] # list of states Vertical-Down

            # 1 encoder per swipe direction
            t0_sp = time.time()
            # H --> this might be vertical
            for n in range(x_fra-W+1):
                inputData = inputFrame[:,n:(n + W),:]
                optimizer_sp.zero_grad() #if its the same encoder should we put it in the end point?
                output = Encoder_sp.forward(inputData).unsqueeze(0)
                c_list_H.append(output)
                if n==0:
                    c_list_H += (W-1)*[c_list_H[n]]
            ##COULD BE WRONG, IT'S NOT PREDICTING BUT GIVING A C TO A GROUP OF INPUTS -
            # MAYBE IT'S ENOUGH BY DECODING USING FRA (NOT FRA+PRE)

            # HI
            for n in range(x_fra-W+1):
                idx = x_fra - np.linspace(1, W, num=W) - n # review
                inputData = inputFrame[:,idx,:]
                optimizer_sp.zero_grad()
                output = Encoder_sp.forward(inputData).unsqueeze(0)
                c_list_HI.append(output)
                if n==0:
                    c_list_HI += (W - 1) * [c_list_HI[n]]

            # V
            for n in range(y_fra - W + 1):
                inputData = inputFrame.permute(0,2,1)[:, n:(n + W), :]
                optimizer_sp.zero_grad()
                output = Encoder_sp.forward(inputData).unsqueeze(0)
                c_list_V.append(output)
                if n == 0:
                    c_list_V += (W - 1) * [c_list_V[n]]

            # VI
            for n in range(y_fra - W + 1):
                idx = y_fra - np.linspace(1, W, num=W) - n  # review
                inputData = inputFrame.permute(0,2,1)[:, idx, :]
                optimizer_sp.zero_grad()
                output = Encoder_sp.forward(inputData).unsqueeze(0)
                c_list_VI.append(output)
                if n == 0:
                    c_list_VI += (W - 1) * [c_list_VI[n]]  # replicate first c values -- check it's actually doing this

            # Flatten stack and transpose, new t will be appent.
            # Transpose again, perform model with c prediction, combine and decode.

            c_H = torch.Tensor(x_fra, 2, CLen, y_fra).cuda(gpu_id) # (128,2,161,160)
            torch.cat(c_list_H, out=c_H)
            C[0, :, :] = c_H.permute(1, 2, 0, 3).reshape(-1, x_fra * y_fra * CLen).unsqueeze(0) # (2,161,128,160) --> (1, 2, 161*128*160)

            c_HI = torch.Tensor(x_fra, 2, CLen, y_fra).cuda(gpu_id)
            torch.cat(c_list_HI, out=c_HI)
            C[1, :, :] = c_HI.permute(1, 2, 0, 3).reshape(-1, x_fra * y_fra * CLen).unsqueeze(0)

            c_V = torch.Tensor(y_fra, 2, CLen, x_fra).cuda(gpu_id)
            torch.cat(c_list_V, out=c_V)
            C[2, :, :] = c_V.permute(1, 2, 0, 3).reshape(-1, x_fra * y_fra * CLen).unsqueeze(0)

            c_VI = torch.Tensor(y_fra, 2, CLen, x_fra).cuda(gpu_id)
            torch.cat(c_list_VI, out=c_VI)
            C[3, :, :] = c_VI.permute(1, 2, 0, 3).reshape(-1, x_fra * y_fra * CLen).unsqueeze(0)

            C = C.permute(1,0,2).reshape(-1, x_fra * y_fra * CLen * 4).unsqueeze(0)
            C_t.append(C)
            print(len(C_t))

            print('Time for spacial encoding in time T: ', time.time() - t0_sp)


        ## REGULAR DYAN
        Ctens_t = torch.Tensor(FRA, 2, CLen*x_fra*y_fra*4).cuda(gpu_id)
        torch.cat(C_t, out=Ctens_t)

        print(Ctens_t.shape, type(Ctens_t))
        Cpred = model_ti.forward(Ctens_t.permute(1,0,2))

        print('Time per forward: ', time.time() - t0_epoch)

        ## GPU LACKS MEMORY - TOO MUCH TIME --> ENCODERS COULD BE PARALLELIZED
        sys.exit()




    loss_val = np.mean(np.array(loss_value))

    ## Checkpoint + prints
    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| time per epoch: %.4f' % (time.time()-t0_epoch))
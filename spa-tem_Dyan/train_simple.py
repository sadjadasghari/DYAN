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
# from DyanOF import OFModel
# from DyanOF import Encoder
# from DyanOF import Decoder
from DyanOFST import OFModel


############################# Import Section #################################



## HyperParameters for the Network
NumOfPoles = 40
CLen = NumOfPoles*4 + 1
EPOCH = 150
BATCH_SIZE = 1
LR = 0.0015
gpu_id = 1
gpu_id2 = 3 # parallelize computation

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
checkptname = "Kitti_simple-ST_lam0.1_"



## Load input data

rootDir = '/home/armandcomas/datasets/Kitti_Flows/'

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


# ## Create the time model
# model_ti = OFModel(Drr, Dtheta, T, PRE, gpu_id)
# model_ti.cuda(gpu_id)
# optimizer = torch.optim.Adam(model_ti.parameters(), lr=LR)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1) # if Kitti: milestones=[100,150]
# loss_mse = nn.MSELoss()
# start_epoch = 1
# ## Create the spatial model X
# Encoder_spX = Encoder(Drr, Dtheta, y_fra, gpu_id) # change by full model?
# Encoder_spX.cuda(gpu_id) #parallelize it?
# optimizer_spX = torch.optim.Adam(Encoder_spX.parameters(), lr=LR)
# scheduler_spX = lr_scheduler.MultiStepLR(optimizer_spX, milestones=[50,100], gamma=0.1) # Parameters?
# ## Create the spatial model Y
# Encoder_spY = Encoder(Drr, Dtheta, x_fra, gpu_id)
# Encoder_spY.cuda(gpu_id) #parallelize it?
# optimizer_spY = torch.optim.Adam(Encoder_spY.parameters(), lr=LR)
# scheduler_spY = lr_scheduler.MultiStepLR(optimizer_spY, milestones=[50,100], gamma=0.1) # Parameters?


# Create the model
model = OFModel(Drr, Dtheta, T, PRE, x_fra, y_fra) # gpu_id2 for parallel computation
model = nn.DataParallel(model, device_ids=[gpu_id, gpu_id2])
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1) # if Kitti: milestones=[100,150]
loss_mse = nn.MSELoss()
start_epoch = 1

## If want to continue training from a checkpoint
if(load_ckpt):
    loadedcheckpoint = torch.load(ckpt_file)
    start_epoch = loadedcheckpoint['epoch']
    model.load_state_dict(loadedcheckpoint['state_dict'])
    optimizer.load_state_dict(loadedcheckpoint['optimizer'])


print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()

#Below, the previous version with separate encoder/decoder
count = 0
# revX_idx = np.linspace(128, 1, num=128)-1
# revY_idx = np.linspace(160, 1, num=160)-1

# ## Start the Training
# for epoch in range(start_epoch, EPOCH+1):
#     t0_epoch = time.time()
#
#     loss_value_T = []
#     scheduler.step()
#
#     for i_batch, sample in enumerate(dataloader):
#
#         data = sample['frames'].squeeze(0).cuda(gpu_id)
#         expectedOut = Variable(data)
#         inputData = Variable(data[:,0:FRA,:])
#
#         # optimizer.zero_grad()
#         # output = model.forward(inputData)
#         #
#         # loss = loss_mse(output[:,FRA], expectedOut[:,FRA]) # if Kitti: loss = loss_mse(output, expectedOut)
#         # loss.backward()
#         # optimizer.step()
#         # loss_value_T.append(loss.data.item())
#
#
#         ##__________Spatial__________
#         # to concatenate layers (encoders) is it enough propagating the loss backwards after concatenating in time?
#         # is it needed to modify the model?
#         out = torch.empty(2, FRA, (x_fra+y_fra)*CLen).cuda(gpu_id)
#         for i in range(FRA):
#             # along columns
#             inputFrame = Variable(data[:,i,:].reshape(-1,x_fra,y_fra))
#
#             # 1 encoder per swipe direction
#             t0_sp = time.time()
#
#             inputData = inputFrame.permute(0, 2, 1)
#             optimizer_spX.zero_grad()
#             outputX = Encoder_spX.forward(inputData).reshape(-1, CLen * x_fra)
#
#             inputData = inputFrame
#             optimizer_spY.zero_grad() #if its the same encoder should we put it in the end point?
#             outputY = Encoder_spY.forward(inputData).reshape(-1, CLen * y_fra)
#
#
#             output = torch.cat((outputX, outputY), 1)
#
#             out[:, i, :] = output
#
#             # Flatten stack and transpose, new t will be appent.
#             # Transpose again, perform model with c prediction, combine and decode.
#
#             print('Time for spacial encoding in time T: ', time.time() - t0_sp)
#
#         ## REGULAR DYAN
#         Cpred = model_ti.forward(out)
#
#         Cpred_X = Cpred[:,9,0:(CLen*x_fra)].reshape(2,CLen,x_fra)
#         Cpred_Y = Cpred[:,9,(CLen*x_fra):].reshape(2,CLen,y_fra)
#
#
#
#         ## Decode
#         Decoder_spX = Decoder(Encoder_spX.rr, Encoder_spX.theta, Encoder_spX.T, 0, gpu_id)
#         Decoder_spY = Decoder(Encoder_spY.rr, Encoder_spY.theta, Encoder_spY.T, 0, gpu_id)
#
#         outX = Decoder_spX.forward(Cpred_X).permute(0,2,1)
#         outY = Decoder_spY.forward(Cpred_Y)
#
#         pred = ((outX+outY) / 2).reshape(-1, x_fra * y_fra)
#
#         print('Time per forward: ', time.time() - t0_epoch)
#
#
#         loss = loss_mse(pred, expectedOut[:,FRA]) # if Kitti: loss = loss_mse(output, expectedOut)
#         loss.backward()
#         optimizer.step()
#         loss_value.append(loss.data.item())
#
#     loss_val = np.mean(np.array(loss_value))
#
#     ## Checkpoint + prints
#     if epoch % saveEvery ==0 :
#         save_checkpoint({	'epoch': epoch + 1,
#                             'state_dict': model.state_dict(),
#                             'optimizer' : optimizer.state_dict(),
#                             },checkptname+str(epoch)+'.pth')
#     print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| time per epoch: %.4f' % (time.time()-t0_epoch))

## Start the Training
for epoch in range(start_epoch, EPOCH+1):
    t0_epoch = time.time()

    loss_value = []
    scheduler.step()

    for i_batch, sample in enumerate(dataloader):

        data = sample['frames'].squeeze(0).cuda(gpu_id)
        expectedOut = Variable(data)
        # inputData = Variable(data[:,0:FRA,:])


        ##__________Encode_Spatial__________
        out = torch.empty(2, FRA, (x_fra+y_fra)*CLen).cuda(gpu_id)

        for i in range(FRA):
            t0_sp = time.time()

            inputFrame = Variable(data[:,i,:].view(-1,x_fra,y_fra))
            optimizer.zero_grad()

            outputH,outputV = model.module.forwardE(inputFrame.permute(0,2,1),inputFrame)
            output = torch.cat((outputH.view(-1, CLen * x_fra), outputV.view(-1, CLen * y_fra)), 1)
            out[:, i, :] = output

            # print('Time for spacial encoding in time T: ', time.time() - t0_sp)

        ##____________Temporal____________
        cPred = model.module.forward(out)

        cPredH = cPred[:, FRA, 0:(CLen * x_fra)].view(2, CLen, x_fra)
        cPredV = cPred[:, FRA, (CLen * x_fra): ].view(2, CLen, y_fra)

        ##__________Decode_Spatial__________

        outH, outV = model.module.forwardD(cPredH, cPredV)
        pred = ((outH.permute(0, 2, 1) + outV) / 2).view(-1, x_fra * y_fra)

        # print('Time per forward: ', time.time() - t0_epoch)

        loss = loss_mse(pred, expectedOut[:,FRA]) # if Kitti: loss = loss_mse(output, expectedOut)
        loss.backward()
        optimizer.step()
        loss_value.append(loss.data.item())

    loss_val = np.mean(np.array(loss_value))

    ## Checkpoint + prints
    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_val, '| time per epoch: %.4f' % (time.time()-t0_epoch))
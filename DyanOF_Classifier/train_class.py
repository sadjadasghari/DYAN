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
import numpy as np
import pandas as pd
from PIL import Image
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
from DyanOF_Class import FullOFModel as FOFModel #faltara importar també el Encoder per congelar els pesos?

############################# Import Section #################################



## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
BATCH_SIZE = 1
LR = 0.0015
gpu_id = 1  # 3?

## For training UCF
# Input -  3 Optical Flow
# Output - 1 Optical Flow
## For training Kitti
# Input -  9 Optical Flow
# Output - 1 Optical Flow

FRA = 9 # input number of frame
PRE = 1 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA # number of row in dictionary(same as input number of frame)
saveEvery = 100

#mnist
x_fra = 64
y_fra = 64


## Load saved model
load_ckpt = True
ckpt_file = 'MS_Model_4px24.pth' # for Kitti Dataset: 'KittiModel.pth'
# checkptname = "UCFModel"
checkptname = "MSModel_Classifier"

#Classes (targets)
c_mov = ['h', 'v']
c_app = ['0', '3']

## Load input data

# set train list name:
trainFolderFile = './datasets/DisentanglingMotion/importing_data/moving_symbols/MovingSymbols2_trainlist.txt'
# trainFolderFile = 'trainlist01.txt'

# set training data directory:
rootDir = './datasets/DisentanglingMotion/importing_data/moving_symbols/output/MovingSymbols2_same_4px-Frames/train'
# rootDir = './datasets/UCF-101-Frames'

trainFoldeList = getListOfFolders(trainFolderFile)[::10]
# if Kitti dataset: use listOfFolders instead of trainFoldeList
# listOfFolders = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]


trainingData = videoDataset(folderList=trainFoldeList,
                            rootDir=rootDir,
                            N_FRAME=N_FRAME)

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

## Create the model for DYAN + classifier
model = TOFModel(Drr, Dtheta, T, PRE, gpu_id)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.l2.parameters(), lr=LR) #SGD?
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1) # if Kitti: milestones=[100,150]
loss_ce = nn.CrossEntropyLoss()
start_epoch = 1


## If want to continue training from a checkpoint
if(load_ckpt):
    loadedcheckpoint = torch.load(ckpt_file)
    # start_epoch = loadedcheckpoint['epoch']
    model.l1.load_state_dict(loadedcheckpoint['state_dict'])
    # optimizer.load_state_dict(loadedcheckpoint['optimizer'])

    for param in model.l1.parameters():
        param.requires_grad = False


print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()


for epoch in range(start_epoch, EPOCH+1):
    loss_value = []
    scheduler.step()
    for i_batch, (sample, target) in enumerate(dataloader): # Added target
        data = sample['frames'].squeeze(0).cuda(gpu_id) #['frames'] vé de getItem a utils.py
        inputData = Variable(data[:,0:FRA,:])
        optimizer.zero_grad()
        output = model.forward(inputData) # forward for frozen DYAN + classifier from flatten c vectors

        loss = loss_ce(output, target) # if Kitti: loss = loss_mse(output, expectedOut)
        loss.backward()
        optimizer.step()
        loss_value.append(loss.data.item())

    loss_val = np.mean(np.array(loss_value))
    # print accuracy also



    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')






    # if epoch % 4 == 0:
    #     print(model.state_dict()['l1.rr'])
    #     print(model.state_dict()['l1.theta'])
    #     # loss_val = float(loss_val/i_batch)
    # print('Epoch: ', epoch, '| train loss: %.4f' % loss_val)
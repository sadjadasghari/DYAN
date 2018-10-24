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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import hsv_to_rgb
from pylab import imshow, show, get_cmap

## Generic imports
import os
import time
import numpy as np
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


## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
BATCH_SIZE = 1
LR = 0.0015
gpu_id = 1 # 3?

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
saveEvery = 2
N_FRAME_FOLDER = 18

#mnist
# x_fra = 64
# y_fra = 64

## Load saved model
load_ckpt = False
ckpt_file = 'MS_Model_4px_22.pth' # for Kitti Dataset: 'KittiModel.pth'
# checkptname = "UCFModel"
checkptname = "Kitti_GL_"

# set training data directory:
rootDir = '/home/armandcomas/DYAN/Code/datasets/Kitti_Flows/'
# rootDir = './datasets/UCF-101-Frames'

#trainFoldeList = getListOfFolders(trainFolderFile)[::10]
# if Kitti dataset: use listOfFolders instead of trainFoldeList
trainFoldeList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, name))]

trainingData = videoDataset(listOfFolders=trainFoldeList,
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

## Create the model
model = OFModel(Drr, Dtheta, T, PRE, gpu_id)
model.cuda(gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1) # if Kitti: milestones=[100,150], UCF [50,100]
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

count = 0
## Start the Training

for epoch in range(start_epoch, EPOCH+1):
    loss_value = []
    #exp_lr_scheduler.step()
    t0t= time.time()
    # for i_batch, sample in enumerate(dataloader):
    #     t1t = time.time()
    #     dataBatch = sample['frames'].squeeze(0)
    #     numBatches = dataBatch.shape[0]
    #     los_val = []
    #     for batchnum in range(numBatches):
    #         data = dataBatch[batchnum].cuda(gpu_id)
    #         expectedOut = data
    #         print(data.shape)
    #         inputData = data[:,0:9,:]
    #         # print(torch.max(inputData),torch.min(inputData))
    #         # inputData = inputData
    #         # print(inputData.shape)
    #         optimizer.zero_grad()
    #         print (inputData.shape)
    #         output = model.forward(inputData)
    #
    #         # torchvision.utils.save_image(output[:, FRA].view(2, x_sz, y_sz), 'predicted_output.png', )
    #         # torchvision.utils.save_image(expectedOut[:, FRA].view(2, x_sz, y_sz), 'expected_output.png', )
    #
    #         print(output.shape, expectedOut.shape)
    #         loss = loss_mse(output, expectedOut)
    #         loss.backward()
    #         optimizer.step()
    #         los_val.append(loss.data[0])
    #     loss_value.append(np.mean(np.array(los_val)))
    #     print('Time per databatch: ', time.time()-t1t)
    # loss_val = np.mean(np.array(loss_value))

    for i_batch, sample in enumerate(dataloader): # 57 samples in dataloader - 1 min/sample
        t1t = time.time()
        los_val = []

        data = sample['frames'].squeeze(0).cuda(gpu_id)
        expectedOut = Variable(data[0,:,:,:])
        inputData = Variable(data[0,:,0:9,:])
        optimizer.zero_grad()
        output = model.forward(inputData)
        # print('exp out vs out: ',expectedOut.shape, output.shape)
        loss = loss_mse(output, expectedOut)
        loss.backward()
        optimizer.step()
        los_val.append(loss.data[0])

        # print('Time per databatch: ', time.time()-t1t)
    loss_value = np.mean(np.array(los_val))

    print('Time per epoch: ', time.time()-t0t)
    if epoch % saveEvery ==0 :
        save_checkpoint({	'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            },checkptname+str(epoch)+'.pth')

    # if epoch % 4 == 0:
    #     print(model.state_dict()['l1.rr'])
    #     print(model.state_dict()['l1.theta'])
    #     # loss_val = float(loss_val/i_batch)
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_value)
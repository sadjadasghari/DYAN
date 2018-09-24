############################# Import Section #################################
## Generic imports
import os
import time
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
## Imports related to PyTorch
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
############################# Import Section #################################

## Dataloader for PyTorch.
class videoDataset(Dataset):
    """Dataset Class for Loading Video"""
    def __init__(self, folderList, rootDir, N_FRAME):

        """
        Args:
            N_FRAME (int) : Number of frames to be loaded
            rootDir (string): Directory with all the Frames/Videoes.
            Image Size = 240,320
            2 channels : U and V
        """
        self.listOfFolders = folderList
        self.rootDir = rootDir
        self.nfra = N_FRAME
        # self.numpixels = 240*320 # If Kitti dataset, self.numpixels = 128*160
        self.numpixels = 64*64 # MNIST moving symbols dataset

    def __len__(self):
        return len(self.listOfFolders)


    # def readData(self, folderName):
    #     path = os.path.join(self.rootDir,folderName)
    #     OF = torch.FloatTensor(2,self.nfra,self.numpixels)
    #     for framenum in range(self.nfra):
    #         flow = np.load(os.path.join(path,str(framenum)+'.npy'))
    #         flow = np.transpose(flow,(2,0,1))
    #         OF[:,framenum] = torch.from_numpy(flow.reshape(2,self.numpixels)).type(torch.FloatTensor)
    #     return OF



    # ____FOR GRAYSCALE IMAGES____
    def readData(self, folderName):
        path = os.path.join(self.rootDir, folderName)
        Fr = torch.FloatTensor(1, self.nfra, self.numpixels)
        for framenum in range(self.nfra):
            # frame = Image.open(os.path.join(path, str(framenum) + '.jpg'))
            frame = cv2.imread(os.path.join(path, str(framenum) + '.jpg'), 0)
            flow = np.array(frame, dtype='uint8')/255.
            flow = np.expand_dims(flow, axis=2)
            flow = np.transpose(flow, (2, 0, 1))
            Fr[:, framenum] = torch.from_numpy(flow.reshape(1, self.numpixels)).type(torch.FloatTensor)
        return Fr

    # def readData(self, folderName):
    #     transform = transforms.Compose([transforms.ToTensor()])
    #     sample = torch.FloatTensor(self.nfra, 64, 64)
    #
    #     value = self.index[folderName]
    #     nFrames = value['last'] - value['first']
    #     numBatches = min(int(nFrames / self.nfra), 1)
    #     sample = torch.FloatTensor(numBatches, self.nfra, self.numpixels)
    #     for batchnum in range(numBatches):
    #         for framenum in range(value['first'], value['first'] + self.nfra):
    #             img = self.imageArray[framenum + self.nfra * batchnum]
    #             # img =  np.transpose(img1,(2,0,1))
    #             img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    #             sample[batchnum, framenum - value['first'], :] = torch.from_numpy(img.reshape(self.nfra)).type(
    #                 torch.FloatTensor)
    #
    #     return sample

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame = self.readData(folderName)
        sample = { 'frames': Frame }

        return sample


## Design poles


def gridRing(N):
    epsilon_low = 0.25
    epsilon_high = 0.15
    rmin = (1-epsilon_low)
    rmax = (1+epsilon_high)
    thetaMin = 0.001
    thetaMax = np.pi/2 - 0.001
    delta = 0.001
    Npole = int(N/4)
    Pool = generateGridPoles(delta,rmin,rmax,thetaMin,thetaMax)
    M = len(Pool)
    idx = random.sample(range(0, M), Npole)
    P = Pool[idx]
    Pall = np.concatenate((P,-P, np.conjugate(P),np.conjugate(-P)),axis = 0)

    return P,Pall

## Generate the grid on poles
def generateGridPoles(delta,rmin,rmax,thetaMin,thetaMax):
    rmin2 = pow(rmin,2)
    rmax2 = pow(rmax,2)
    xv = np.arange(-rmax,rmax,delta)
    x,y = np.meshgrid(xv,xv,sparse = False)
    mask = np.logical_and( np.logical_and(x**2 + y**2 >= rmin2 , x**2 + y **2 <= rmax2),
                           np.logical_and(np.angle(x+1j*y)>=thetaMin, np.angle(x+1j*y)<=thetaMax ))
    px = x[mask]
    py = y[mask]
    P = px + 1j*py

    return P


# Create Gamma for Fista
def getWeights(Pall,N):
    g2 = pow(abs(Pall),2)
    g2N = np.power(g2,N)

    GNum = 1-g2
    GDen = 1-g2N
    idx = np.where(GNum == 0)[0]

    GNum[idx] = N
    GDen[idx] = pow(N,2)
    G = np.sqrt(GNum/GDen)
    return np.concatenate((np.array([1]),G))

## Functions for printing time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

## Function to save the checkpoint
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def getListOfFolders(File):
    data = pd.read_csv(File, sep=" ", header=None)[0]
    # data = data.str.split('/',expand=True)[1]
    data = data.str.rstrip(".avi").values.tolist()

    return data

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
from numpy.lib.stride_tricks import as_strided
## Imports related to PyTorch
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
############################# Import Section #################################


class videoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, listOfFolders, rootDir):

        """
        Args:
            N_FRAME (int) : Number of frames to be loaded
            rootDir (string): Directory with all the Frames/Videoes.
        """
        self.listOfFolders = listOfFolders
        self.rootDir = rootDir


    def __len__(self):
        return len(self.listOfFolders)

    def readData(self, folderName):
        path = os.path.join(self.rootDir,folderName)
        numBatches = 1#min(int(nFrames/nfra),100)
        sample = torch.FloatTensor(numBatches,2,10,128*160)
        for batchnum in range(numBatches):
            for framenum in range(10):

                flow = np.load(os.path.join(path,str(framenum)+'.npy'))
                flow = np.transpose(flow,(2,0,1))
                sample[batchnum,:,framenum,:] = torch.from_numpy(flow.reshape(2,128*160)).type(torch.FloatTensor)

        return sample

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        frames = self.readData(folderName)
        sample = { 'frames': frames }

        return sample
## Dataloader for PyTorch.
# class videoDataset(Dataset):
#     """Dataset Class for Loading Video"""
#     def __init__(self, folderList, rootDir, N_FRAME, N_FRAME_FOLDER): #N_FRAME = FRA+PRE
#
#         """
#         Args:
#             N_FRAME (int) : Number of frames to be loaded
#             rootDir (string): Directory with all the Frames/Videoes.
#             Image Size = 240,320
#             2 channels : U and V
#         """
#         self.listOfFolders = folderList
#         self.rootDir = rootDir
#         self.nfra = N_FRAME
#         self.nfrafol = N_FRAME_FOLDER
#         # self.numpixels = 240*320 # If Kitti dataset, self.numpixels = 128*160
#         self.numpixels = 128*160
# #         self.numpixels = 64*64 # MNIST moving symbols dataset
#
#     def __len__(self):
#         return len(self.listOfFolders)
#
#
#     # def readData(self, folderName):
#     #     path = os.path.join(self.rootDir,folderName)
#     #     OF = torch.FloatTensor(2, self.nfra, self.numpixels)
#     #     for framenum in range(self.nfra):
#     #         flow = np.load(os.path.join(path,str(framenum)+'.npy'))
#     #         flow = np.transpose(flow,(2,0,1))
#     #         OF[:,framenum] = torch.from_numpy(flow.reshape(2,self.numpixels)).type(torch.FloatTensor)
#     #     return OF
#
#     def readData(self, folderName):
#         path = os.path.join(self.rootDir,folderName)
#         OF = torch.FloatTensor(2,self.nfrafol,self.numpixels)
#
#         for framenum in range(self.nfrafol):
#             flow = np.load(os.path.join(path,str(framenum)+'.npy'))
#             flow = np.transpose(flow,(2,0,1))
#             OF[:,framenum] = torch.from_numpy(flow.reshape(2,self.numpixels)).type(torch.FloatTensor)
#         return OF
#
#
#     def __getitem__(self, idx):
#         folderName = self.listOfFolders[idx]
#         Frame = self.readData(folderName)
#         sample = { 'frames': Frame }
#
#         return sample


def getCells_slide(inp, ws):
    xs = inp.shape[0]
    ys = inp.shape[1]
    # nchan = inp.shape[0]
    nchan = 1
    cells = []
    for n in range(nchan):
        for i in range(xs):
            for j in range(ys):
                cells.append(cell_neighbors(inp,i,j,ws))

    #Still to decide how do we treat the edge pixels (which can't have a full window)
        #- we could obtain the c's computed from the computation of the neighboring pixels
        #- we could pad the frame (mirroring would be best)

    return cells


def getCells_stride_Kitti(inp, ws):
    xs = inp.shape[0]
    ys = inp.shape[1]
    # nchan = inp.shape[0]
    cells = []
    for j in range(2,ys,5):
        for i in range(0, xs, 5):
            cells.append(cell_neighbors(inp,i,j,ws))

    #Still to decide how do we treat the edge pixels (which can't have a full window)
        #- we could obtain the c's computed from the computation of the neighboring pixels
        #- we could pad the frame (mirroring would be best)

    return cells

def sliding_window(arr, window_size):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)

def cell_neighbors(arr, i, j, d):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d+1)
    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)
    
    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()
    
    

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
    data = data.str.split('/',expand=True)[1] # --> This is commented for the warp.py!!! review for other scripts
    data = data.str.rstrip(".avi").values.tolist()

    return data

def getListOfFolders_warp(File):
    data = pd.read_csv(File, sep=" ", header=None)[0]
    data = data.str.rstrip(".avi").values.tolist()

    return data

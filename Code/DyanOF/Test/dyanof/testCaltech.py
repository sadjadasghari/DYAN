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
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from skimage import io, transform
from matplotlib.backends.backend_pdf import PdfPages

## Dependencies classes and functions
from utils import PSNR
from utils import SHARP
from utils import asMinutes
from utils import timeSince
from utils import getListOfFolders
from utils import creatRealDictionary
from sc2layerModel import SClayer
from skimage import measure


from scipy.misc import imread, imresize
from skimage.measure import compare_mse as mse
from collections import defaultdict
import matplotlib
import scipy.io as sio
import cv2
import seaborn as sns
from matplotlib.colors import hsv_to_rgb
from sklearn import preprocessing
############################# Import Section #################################
import math
from pylab import imshow, show, get_cmap
import imutils
import cv2
import seaborn as sns
import scipy
import math
start = time.time()
import pyflow
# Hyper Parameters
FRA = 9
PRE = 1
N_FRAME = FRA+PRE
T = FRA


def loadModel(ckpt_file):
	loadedcheckpoint = torch.load(ckpt_file)#,map_location={'cuda:1':'cuda:0'})
	stateDict = loadedcheckpoint['state_dict']
	
	# load parameters
	Dtheta = stateDict['l1.theta'] 
	Drr    = stateDict['l1.rr']
	Gamma  = stateDict['l1.gamma']
	model = SClayer(Drr, Dtheta, Gamma, T)
	model.cuda(gpu_id)
	Drr = Variable(Drr.cuda(gpu_id))
	
	Dtheta = Variable(Dtheta.cuda(gpu_id))
	dictionary = creatRealDictionary(N_FRAME,Drr,Dtheta)
	
	return model, dictionary

def process_im(im, desired_sz=(128, 160)):
	target_ds = float(desired_sz[0])/im.shape[0]
	im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
	d = int((im.shape[1] - desired_sz[1]) / 2)
	im = im[:, d:d+desired_sz[1]]
	# im = imutils.resize(im, width=160,height=128)
	return im


def SSIM(predi,pix):
	pix = pix.astype(float)
	predict = predi.astype(float)
	ssim_score = measure.compare_ssim(pix, predict, win_size=11, data_range = 1.,multichannel=True,
					gaussian_weights=True,sigma = 1.5,use_sample_covariance=False,
					K1=0.01,K2=0.03)

	return ssim_score


def scipwarp(img, u, v):
	M, N, _ = img.shape
	x = np.linspace(0,N-1, N)
	y = np.linspace(0,M-1, M)
	x, y = np.meshgrid(x, y)
	x += u
	y += v
	warped = img
	warped[:,:,0] = scipy.ndimage.map_coordinates(img[:,:,0], [y.ravel(),x.ravel()], order=1, mode='nearest').reshape(img.shape[0],img.shape[1])
	warped[:,:,1] = scipy.ndimage.map_coordinates(img[:,:,1], [y.ravel(),x.ravel()], order=1, mode='nearest').reshape(img.shape[0],img.shape[1])
	warped[:,:,2] = scipy.ndimage.map_coordinates(img[:,:,2], [y.ravel(),x.ravel()], order=1, mode='nearest').reshape(img.shape[0],img.shape[1])
	return warped

# from utils import gridRing
# P,Pall = gridRing(N)
# Drr = abs(P)
# # Drr = torch.from_numpy(Drr).float()
# Dtheta = np.angle(P)

alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0



gpu_id = 0

ckpt_file = 'NormDict34.pth'

rootDir = '/home/abhishek/Caltech/images'
# folder = 'set07V007/'
# folder = 'set08V008/'
# folder = 'set10V011/'

## Load model from a checkpoint file
model,  dictionary = loadModel(ckpt_file)
folderList = ['set10V011']
__imgsize__ = (128,160)
mse = []
ssim = []
psnr = []
sample = torch.FloatTensor(2,FRA,128*160)

# folderList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
# folderList.sort()
for folder in folderList:
	print(folder)
	frames = [each for each in os.listdir(os.path.join(rootDir, folder)) if each.endswith(('.jpg','.jpeg','.bmp','png'))]
	frames.sort()
	# print(folder)
	for i in range(1,len(frames)-11,10):
		print(i)
		for ii in range(9):
			# print("Inside Main", ii+i)
			imgname = os.path.join(rootDir,folder,frames[ii+i])
			img = Image.open(imgname)
			img1 = process_im(np.array(img))/255.
			# img1 = img1.astype(float) / 255.
			# print(img1.shape)

			imgname = os.path.join(rootDir,folder,frames[ii+1+i])
			img = Image.open(imgname)
			img2 = process_im(np.array(img))/255.
			# img2 = img2.astype(float) / 255.
			# print(img2.shape)
			u, v,_ = pyflow.coarse2fine_flow( img2, img1, alpha, ratio, minWidth, 
					nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
			flow = np.concatenate((u[..., None], v[..., None]), axis=2)
			flow = np.transpose(flow,(2,0,1))
			sample[:,ii,:] = torch.from_numpy(flow.reshape(2,128*160)).type(torch.FloatTensor)

		# print("Original",i+ii+2)
		imgname = os.path.join(rootDir,folder,frames[i+ii+2])
		img = Image.open(imgname)
		original = process_im(np.array(img))/255.
		# print("Tenth",i+ii+1)
		imgname = os.path.join(rootDir,folder,frames[i+ii+1])
		img = Image.open(imgname)
		tenth = process_im(np.array(img))/255.
		
		inputData = sample.cuda()
		with torch.no_grad():	
			sparse = model.forward(Variable(inputData))
		prediction = torch.matmul(dictionary,sparse)[:,FRA,:].data.permute(1,0).resize(128,160,2).cpu().numpy()
		img_back = scipwarp(tenth,prediction[:,:,0],prediction[:,:,1])
		img_back = np.clip(img_back, 0, 1.)
		# print(img_back.shape)
		# img_back = np.uint8(img_back)
		# plt.imshow(img_back)
		# plt.show()
		#if i == 111:
		ax = plt.subplot(2,1,1)
		ax.imshow(original)
		ax.set_title("Original")

		ax = plt.subplot(2,1,2)
		ax.imshow(img_back)
		ax.set_title("10th Warped using predicted OF")
		plt.tight_layout()
		plt.savefig("Results/"+str(i+FRA+1)+".png")
		
		meanserror = np.mean( (img_back - original) ** 2 )
		mse.append(meanserror)
		peaksnr = 10*math.log10(1./meanserror)
		psnr.append(peaksnr)	
		ssim.append(SSIM(original, img_back))

print("PSNR : ", np.mean(np.array(psnr)))
print("MSE : ", np.mean(np.array(mse)))
print("SSIMs : ", np.mean(np.array(ssim)))

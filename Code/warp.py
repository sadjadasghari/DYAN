import torch
from torch.autograd import Variable

## Generic imports
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from DyanOF import OFModel
from utils import getListOfFolders

from skimage import measure
from scipy.misc import imread, imresize

# Hyper Parameters
FRA = 3  # if Kitti: FRA = 9
PRE = 1  # represents predicting 1 frame
N_FRAME = FRA + PRE
T = FRA
numOfPixels = 64 * 64  # if Kitti: 128*160

gpu_id = 1
opticalflow_ckpt_file = 'MS_Model_4px_4.pth'  # if Kitti: 'KittiModel.pth'


def loadOpticalFlowModel(ckpt_file):
    loadedcheckpoint = torch.load(ckpt_file)
    stateDict = loadedcheckpoint['state_dict']

    # load parameters
    Dtheta = stateDict['l1.theta']
    Drr = stateDict['l1.rr']
    model = OFModel(Drr, Dtheta, FRA, PRE, gpu_id)
    model.cuda(gpu_id)

    return model


def warp(input, tensorFlow):
    torchHorizontal = torch.linspace(-1.0, 1.0, input.size(3))
    torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(input.size(0), 1, input.size(2),
                                                                          input.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
    torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))

    tensorGrid = torch.cat([torchHorizontal, torchVertical], 1).cuda(gpu_id)
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=input, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='border')

ofmodel = loadOpticalFlowModel(opticalflow_ckpt_file)
ofSample = torch.FloatTensor(2, FRA, numOfPixels)
# set test list name:
testFolderFile = './datasets/DisentanglingMotion/importing_data/moving_symbols/MovingSymbols2_testlist.txt'
# set test data directory:
rootDir = './datasets/DisentanglingMotion/importing_data/moving_symbols/output/MovingSymbols2_same_4px-Frames/test'
# for UCF dataset:
testFoldeList = getListOfFolders(testFolderFile)[::10]
## if Kitti: use folderList instead of testFoldeList
## folderList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
## folderList.sort()

# flowDir = '/home/abhishek/Workspace/UCF_Flows/Flows_ByName/'
flowDir = './datasets/DisentanglingMotion/importing_data/moving_symbols/output/MovingSymbols2_same_4px-OF/test'

for	numfo,folder in enumerate(testFoldeList):
    print("Started testing for - "+ folder)

    if not os.path.exists(os.path.join("Results", str(10*numfo+1))):
        os.makedirs(os.path.join("Results", str(10*numfo+1)))

    frames = [each for each in os.listdir(os.path.join(rootDir, folder)) if each.endswith(('.jpg'))]
    frames.sort()

    path = os.path.join(rootDir,folder,frames[4])
    img = Image.open(path)
    original = np.array(img)/255.

    path = os.path.join(rootDir,folder,frames[3])
    img = Image.open(path)
    frame4 = np.array(img)/255.

    tensorinput = torch.from_numpy(frame4).type(torch.FloatTensor).permute(2,0,1).cuda(gpu_id).unsqueeze(0)

    flow_folder = folder.split('/')[1]
    for k in range(3):
        flow = np.load(os.path.join(flowDir,flow_folder,str(k)+'.npy'))
        flow = np.transpose(flow,(2,0,1))
        ofSample[:,k,:] = torch.from_numpy(flow.reshape(2,numOfPixels)).type(torch.FloatTensor)

    ofinputData = ofSample.cuda(gpu_id)

    with torch.no_grad():
        ofprediction = ofmodel.forward(Variable(ofinputData))[:,3,:].data.resize(2,64,64).unsqueeze(0)

    warpedPrediction = warp(tensorinput,ofprediction).squeeze(0).permute(1,2,0).cpu().numpy()
    warpedPrediction = np.clip(warpedPrediction, 0, 1.)


    plt.imsave(os.path.join("Results", str(10*numfo+1),'OriFrame-%04d' % (5)+'.png'), original)
    plt.imsave(os.path.join("Results", str(10*numfo+1),'PDFrame-%04d' % (5)+'.png'), warpedPrediction)
    plt.imsave(os.path.join("Results", str(10*numfo+1), 'Pr-OriFrame-%04d' % (5) + '.png'), frame4)
    plt.close()


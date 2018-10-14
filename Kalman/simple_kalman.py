import sys
## Imports related to PyTorch

import scipy.misc
import scipy.signal

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
import math as mt
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import videoDataset
from utils import save_checkpoint
# from Kalman import KalmanFilter
from os import listdir
from os.path import isfile, join


plot = True

T = 3
N = 100
n_var = 0.4

def KalmanFilter (H, F, x_old, P_old, meas, R, Q):
    # S = H @ P_old @ H.T + R
    # K = P_old @ H.T @ np.linalg.inv(S)
    #
    y_err = (np.matrix(meas) - np.matmul(H, x_old.T))
    S = np.matmul(np.matmul(H,P_old),H.T) + np.matrix(R) # (1,4) (4,4) (4,1) + (1,1)

    K = np.matmul(np.matmul(P_old,H.T),np.linalg.inv(S)) # (4,4) (4,1) (1,1)

    I = F # (4,4)
    x_new = x_old + (K * y_err).T # (1,4) = (1,4) + ((4,1)(1,1)).T

    # print('correction after obs: ',(K * y_err).T)
    P_new = np.matmul((I - np.matmul(K, H)), P_old) # (4,4) = ((4,4) - (4,1)(1,4)) (4,4)

    x_old = x_new # (1,4)

    P_old = np.matmul(np.matmul(F,P_new),F.T) + Q

    return x_old, P_old

Po,Pall = gridRing(4)
poles = [Pall[0], Pall[2]]
print(poles)

Drr = abs(Po)
Dtheta = np.angle(Po)
WVar = []

# generate dictionary
for i in range(0, N):  # matrix 8
    W1 = (Drr**i) * mt.cos(i * Dtheta)
    W2 = (-Drr**i) * mt.cos(i * Dtheta)
    W3 = (Drr**i) * mt.sin(i * Dtheta)
    W4 = (-Drr**i) * mt.sin(i * Dtheta)
    W = np.concatenate((W1, W2, W3, W4), 0)
    W = np.expand_dims(W, axis=0)
    WVar.append(W)
dic = np.concatenate((WVar), 0)

syst = scipy.signal.lti([], poles, 1) #Pall or poles
systD = syst.to_discrete()
print(syst)
ir = scipy.signal.impulse(syst, N=N)
print(ir[1])
# obs_x = ir[0] + 0.1 * np.random.random(N) * ir[0]


ir = -np.flipud(ir[1])
obs_y = ir + np.random.normal(0,n_var,N) * ir
y=obs_y
# print(y[0:3]-ir[0:3])
# dic (40,4)
count = 0
result = []
for n in range(3,N):

    H = dic[n,:]
    meas = y[n]

    if count == 0:
        H = dic[0:3,:]
        x = np.linalg.lstsq(H, ir[0:3], rcond=None)[0]
        # print(np.matmul(H,x), x, y[0:3], H)
        f_meas = np.matmul(H,x)
        # P = np.zeros([4,4]) # inicialize as the cov of X
        H = dic[3, :]
        meas = ir[3]
        R = 0.1 ** 2
        Hinv = np.linalg.lstsq(np.matrix(x), np.matrix(meas), rcond=None)[0]
        Q = np.matmul(np.matmul(Hinv, np.matrix(R)), Hinv.T)
        P = Q
        result = f_meas.tolist()
        count +=1

    F = 0.9*np.eye(4)

    H = np.expand_dims(H, axis=0)

    x,P = KalmanFilter(H, F, x, P, meas, R, Q)

    result.append(np.asscalar(np.matmul(np.expand_dims(dic[n,:],axis=0), x.T)))
# print(result - obs_y)
if plot:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    pl1 = ax.plot(ir, label='expected')
    pl2 = ax.plot(obs_y, label='measured')
    pl3 = ax.plot(result, label='kalman')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    # ax.plt.legend(handles=[pl1, pl2, pl3])
    ax.set_title('Impulse response for given poles')
    plt.savefig('IR2.png')


        # optimizer.zero_grad()
        # loss = loss_mse(output[:,FRA], expectedOut[:,FRA]) # if Kitti: loss = loss_mse(output, expectedOut)
        # loss.backward()
        # optimizer.step()
        # loss_value.append(loss.data.item())

    # loss_val = np.mean(np.array(loss_value))

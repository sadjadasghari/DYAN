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

def readData(rootDir, folderName):
    path = os.path.join(rootDir, folderName)
    OF = 1
    with open(os.path.join(path, "labels.txt")) as f:
        target = [line.split() for line in f]
        target = [target[0][0], int(target[1][0])]
    return OF, target

OF, target = readData('/home/armandcomas/DYAN/Code/datasets/DisentanglingMotion/importing_data/moving_symbols/output/MovingSymbols2_same_4px-OF/train', 'Horizontal_video_1')
print(target)


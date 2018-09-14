############################# Import Section #################################
## Generic imports
import numpy as np
from PIL import Image
import pandas as pd
import os
import cv2

import cv2
print(cv2.__version__)

pathOri = './datasets/UCF-101/'
pathDest = './datasets/UCF-101-Flow/'

# data = pd.read_csv('trainlist01.txt', sep=" ", header=None)[0]
# data = data.str.rstrip(".avi").values.tolist()
#
# for i in data:
#     vid = os.path.join(pathOri, i + '.avi')
#     vidcap = cv2.VideoCapture(vid)
#     success,image = vidcap.read()
#     count = 0
#     success = True
#     while success:
#         if not os.path.exists(os.path.join(pathDest, i)):
#             os.makedirs(os.path.join(pathDest, i))
#         cv2.imwrite(os.path.join(pathDest, i, "%d.jpg" % count) , image)     # save frame as JPEG file
#         success,image = vidcap.read()
#         # print ('Read a new frame: ', success, ' ', os.path.join(pathDest, i, "%d.jpg" % count))
#         count += 1

data = pd.read_csv('testlist01.txt', sep=" ", header=None)[0]
data = data.str.rstrip(".avi").values.tolist()

for i in data:
    vid = os.path.join(pathOri, i + '.avi')
    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        if not os.path.exists(os.path.join(pathDest, i)):
            os.makedirs(os.path.join(pathDest, i))
        cv2.imwrite(os.path.join(pathDest, i, "%d.jpg" % count) , image)     # save frame as JPEG file
        success,image = vidcap.read()
        #print ('Read a new frame: ', success, ' ', os.path.join(pathDest, i, "%d.jpg" % count))
        count += 1
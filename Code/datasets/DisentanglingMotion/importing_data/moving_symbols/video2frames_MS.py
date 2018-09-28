############################# Import Section #################################
## Generic imports
import numpy as np
import sys
from PIL import Image
import pandas as pd
import os
import cv2

import cv2
print(cv2.__version__)

pathOri = './output/MovingSymbols2_same_4px/test/'
pathDest = './output/MovingSymbols2_same_4px-Frames/test/'

data = pd.read_csv('MovingSymbols2_testlist.txt', sep=" ", header=None)[0]

data = data.str.rstrip(".avi").values.tolist()
# print(np.load(pathOri + 'Horizontal/Horizontal_symbol_classes.npy'))


for i in data:

    # obtain labels
    m_label = i.split('/')[0]
    a_labels = np.load(os.path.join(pathOri, m_label, m_label + '_symbol_classes.npy'))
    class_index = i.strip(os.path.join(m_label, m_label + '_video_'))
    a_label = a_labels[int(class_index) - 1]
    a_label = int(a_label)

    labels = [m_label, a_label]

    with open(os.path.join(pathDest, i, 'labels.txt'), 'w') as f:
        for item in labels:
            f.write("%s\n" % item)

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

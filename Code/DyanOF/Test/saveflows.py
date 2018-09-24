import os 
import numpy as np
import hickle as hkl
import pyflow

def getIndex(sources_list):
	index = {}
	folderName = set(sources_list)
	for folder in folderName:
		first = 0
		last = 0
		for count, name in enumerate(sources_list):
			if folder == name:
				if first == 0:
					if count == 0:
						first = count-1
					else:
						first = count
				
				last = count
				index[folder] = {"first":first, "last":last+1}
			else:
				continue

	if index['city-2011_09_26_drive_0001_sync']['first'] == -1:
		index['city-2011_09_26_drive_0001_sync']['first'] = 0

	return index


alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0

rootDir = '/home/abhishek/Workspace/KITTI/prednet_kitti_data'
training_file = 'X_train.hkl'

sources_file = 'sources_train.hkl'

imageArray = hkl.load(os.path.join(rootDir, training_file))
sources_list = hkl.load(os.path.join(rootDir, sources_file))

indexArray = getIndex(sources_list)
listOfFolders = [key for key in indexArray]

for folder in listOfFolders:
	value = indexArray[folder]
	nFrames = value['last'] - value['first']
	numBatches = min(int(nFrames/11),1)
	for batchnum in range(numBatches):
			for framenum in range(10):
				img1 = imageArray[framenum + 11*batchnum]
				img1 = img1.astype(float) / 255.
				img2 = imageArray[framenum + 1 + 11*batchnum]/255.
				img2 = img2.astype(float) / 255.
				u, v,_ = pyflow.coarse2fine_flow( img2, img1, alpha, ratio, minWidth, 
									nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
				flow = np.concatenate((u[..., None], v[..., None]), axis=2)
				if not os.path.exists(os.path.join("Kitti_Flows", folder)):
					os.makedirs(os.path.join("Kitti_Flows", folder))
				np.save(os.path.join("Kitti_Flows", folder,str(framenum)+'.npy'), flow)
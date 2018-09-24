import cv2
import numpy as np
import matplotlib.pyplot as plt

a = cv2.imread('outFlow.png')/255.
print(a.shape)
print(np.max(a), np.min(a))

m,n,_ = a.shape

for i in range(m):
	for j in range(n):
		if a[i,j,0] <0.2 and a[i,j,1] <0.2 and a[i,j,2] <0.2:
			a[i,j,:] = 0
		else:
			a[i,j,:] = 1
plt.imshow(a)
plt.show()


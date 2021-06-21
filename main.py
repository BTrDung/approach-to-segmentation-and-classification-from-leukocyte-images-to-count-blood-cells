import cv2 as cv
import time 

import numpy as np 
from matplotlib import pyplot as plt 

path = 'E:\Github\Machine-learning-for-counting-blood-cells\dataset2-master\images\TRAIN_SIMPLE\EOSINOPHIL\_0_207.jpeg'

img_rgb = cv.imread(path, cv.COLOR_BGR2RGB) 
cv.imshow('image orginal', img_rgb)
# cv.waitKey(0) 
# cv.destroyAllWindows() 

color = ('r', 'g', 'b')
for i, col in enumerate(color): 
    histr = cv.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(histr, color = col) 
    plt.xlim([0, 256]) 
plt.show()

img_hsv     = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
lower_red = np.array([110, 110, 110]) 
# lower_red = np.array([0, 12, 182]) 
upper_red = np.array([200,200,200]) 
# upper_red = np.array([238, 21, 132]) 
mask    = cv.inRange(img_hsv, lower_red, upper_red)
# result  = cv.bitwise_and(img_rgb, img_rgb, mask=mask)
cv.imshow('mask', mask)
# cv.imshow('res', result)

cv.waitKey(0) 
cv.destroyAllWindows() 
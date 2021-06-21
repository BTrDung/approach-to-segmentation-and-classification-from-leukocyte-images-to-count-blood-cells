import cv2 as cv
import time 

import numpy as np 
from matplotlib import pyplot as plt 

def histogram_rgb(img): 
    color = ('r', 'g', 'b')
    for i, col in enumerate(color): 
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color = col) 
        plt.xlim([0, 256]) 
    plt.show()
# -----------------------------------------------------------------
def detect_by_mask(image):
    img = np.copy(image)  
    img_hsv     = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    lower_red   = np.array([5, 5, 5]) 
    upper_red   = np.array([210, 210, 200])  

    mask        = cv.inRange(img_hsv, lower_red, upper_red)
    result      = cv.bitwise_and(img, img, mask=mask)
    # cv.imshow('mask', mask)
    # cv.imshow('res', result)
    # cv.waitKey(0) 
    # cv.destroyAllWindows() 
    return result
# -----------------------------------------------------------------
def detect_edge(image): 
    img = np.copy(image)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    kernels_x = np.array([  [-1, 0, 1],
                            [-2, 0, 2], 
                            [-1, 0, 1]])
    kernels_y = kernels_x.T

    Gx = cv.filter2D(img, -1, kernels_x)
    Gy = cv.filter2D(img, -1, kernels_y)
    G1 = np.sqrt(np.square(Gx * 1.0) + np.square(Gy * 1.0))
    cv.imshow('gg', G1)
    cv.waitKey(0) 
    cv.destroyAllWindows() 
    return None 
# -----------------------------------------------------------------
path    = 'E:\Github\Machine-learning-for-counting-blood-cells\dataset2-master\images\TRAIN_SIMPLE\EOSINOPHIL\_0_207.jpeg'
img_rgb = cv.imread(path, cv.COLOR_BGR2RGB) 
# cv.imshow('image orginal', img_rgb) 
# -----------------------------------------------------------------
new_img = detect_by_mask(img_rgb)
# -----------------------------------------------------------------
detect_edge(new_img)
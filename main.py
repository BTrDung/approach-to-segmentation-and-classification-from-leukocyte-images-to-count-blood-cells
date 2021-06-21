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

def histogram_gray(img): 
    plt.hist(img.flatten(), bins=256, range=(0, 1)) 
    plt.plot() 
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
    kernels_x   = np.array([[-1, 0, 1],
                            [-2, 0, 2], 
                            [-1, 0, 1]])
    kernels_y   = np.array([[1, 2, 1], 
                            [0, 0, 0], 
                            [-1, -2, -1]])
    kernels_g   = np.ones((3, 3), np.float32) / 9

    Gg = cv.filter2D(img, -1, kernels_g)
    img = Gg

    Gx = cv.filter2D(img, -1, kernels_x)
    Gy = cv.filter2D(img, -1, kernels_y)
    G1 = np.sqrt(np.square(Gx * 1.0) + np.square(Gy * 1.0))
    cv.imshow('detect edge', G1)
    cv.waitKey(0) 
    cv.destroyAllWindows() 
    return None 
# -------------------SOURCE-IMAGE---------------------------------
path    = 'E:\Github\Machine-learning-for-counting-blood-cells\dataset2-master\images\TRAIN_SIMPLE\EOSINOPHIL\_0_207.jpeg'
img_rgb = cv.imread(path, cv.COLOR_BGR2RGB) 
# cv.imshow('image orginal', img_rgb) 

# ------------------RGB-TO-GRAY_----------------------------------
img_gray= cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
# cv.imshow('cc1', img_gray)

# -----------------GAUSSIAN-FILTER--------------------------------
# img_gauss_filter = cv.adaptiveThreshold(img_rgb, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2) 
img_gauss_filter = cv.GaussianBlur(img_gray, (5, 5), 0) 
cv.imshow('cc2', img_gauss_filter)
histogram_gray(img_gauss_filter) 

# -----------------OTSU-THRESHOLDING------------------------------
blur = np.copy(img_gauss_filter)
ret3, th3 = cv.threshold(img_gauss_filter,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow('cc3', th3)
cv.waitKey(0) 
cv.destroyAllWindows() 
# -----------------SOBEL-EDGE-DETECTION---------------------------
# Source img 
# Gaussian Filter 
# S chanel 
# Otsu Threshold 
# Sbol Edge Detection
# Hough Transform

import cv2 as cv
import time
import numpy as np
from matplotlib import pyplot as plt
from WBC import count_wbc
# ----------------------------------------------------------------
def histogram_rgb(img):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def prtplt(hist):
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.legend(('histogram'), loc='upper left')
    plt.show()


def histogram_gray(image):
    img = np.copy(image)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    size = img.shape[0] * img.shape[1]
    hist = hist / size
    # prtplt(hist)
    return hist

# ----------------------------------------------------------------
def count_rbc(path, path_input, path_mask, path_output, path_rbc):
    img = cv.imread(path_input) 
    msk = cv.imread(path_mask) 
   
    for i in range(0, img.shape[0]): 
        for j in range(0, img.shape[1]): 
            if msk[i][j][0] == 255 and msk[i][j][1] == 255 and msk[i][j][2] == 255: 
                img[i][j] = msk[i][j] 

    cv.imwrite(path_rbc, img) 
    img_rgb         = cv.imread(path_rbc)
    img_gray        = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY) 
    ret, img_otsu   = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU) 
    img_otsu = cv.Canny(img_otsu, 255, 255)
    

    cv.imwrite(path + 'canny.png', img_otsu)
    gray            = np.copy(img_otsu)
    rows            = gray.shape[0]
    circles         = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, rows / 30 , param1=50, param2=10, minRadius=11, maxRadius=19)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1]) 
            radius = i[2]
            cv.circle(img_rgb, center, radius, (255, 0, 255), 2)
    cv.imwrite(path_output, img_rgb)
    return circles[0].shape[0]

# def file_his(): 
#     img = cv.imread('E:/Github/cells/LISC Database/Main Dataset/Baso/49.bmp') 
#     img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#     ret, img_thres= cv.threshold(img, 155, 255,cv.THRESH_BINARY)
#     cv.imwrite('___.png', img_thres)
# file_his()

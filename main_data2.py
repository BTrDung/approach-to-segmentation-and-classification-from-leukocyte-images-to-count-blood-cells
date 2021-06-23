import cv2 as cv
import time

import numpy as np

from matplotlib import pyplot as plt


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
    return prtplt(hist)

 
 
# -------------------SOURCE-IMAGE---------------------------------
path = 'E:/Github/Machine-learning-for-counting-blood-cells/data2/1.bmp'
img_rgb = cv.imread(path)

# ------------------RGB-TO-GRAY_----------------------------------
img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY) 

# -----------------OTSU-THRESHOLDING------------------------------
ret, img_otsu = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cv.imwrite('E:/Github/Machine-learning-for-counting-blood-cells/data2/otsu.png', img_otsu)

# -----------------HOUGH-CIRCLE-TRANSFORM---------------------------
gray = np.copy(img_otsu)
rows = gray.shape[0]
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 15, param1=350, param2=6, minRadius=11, maxRadius=25)

print(len(circles[0]))

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        cv.circle(img_rgb, center, 1, (0, 100, 100), 3)
        radius = i[2]
        cv.circle(img_rgb, center, radius, (255, 0, 255), 2)

cv.imwrite('E:/Github/Machine-learning-for-counting-blood-cells/data2/output.png', img_rgb)

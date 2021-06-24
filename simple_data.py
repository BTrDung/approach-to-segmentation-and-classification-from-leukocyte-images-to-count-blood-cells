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
  
# -----------------------------------------------------------------
def detect_edge(image):
    img = np.copy(image)
    kernels_x = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    kernels_y = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])

    Gx = cv.filter2D(img, -1, kernels_x)
    Gy = cv.filter2D(img, -1, kernels_y)
    G1 = np.sqrt(np.square(Gx * 1.0) + np.square(Gy * 1.0))
    return G1

# -------------------SOURCE-IMAGE---------------------------------
path            = './data1/1.jpg'
img_rgb         = cv.imread(path)
img_gray        = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
ret, img_otsu   = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
# cv.imwrite('E:\Github\Machine-learning-for-counting-blood-cells\data1\otsu.png', img_otsu)

# -----------------HOUGH-CIRCLE-TRANSFORM---------------------------
gray = np.copy(img_otsu)
rows = gray.shape[0] 
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 98 , param1=500, param2=6, minRadius=11, maxRadius=19)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        cv.circle(img_rgb, center, 1, (0, 100, 100), 3)
        radius = i[2]
        cv.circle(img_rgb, center, radius, (255, 0, 255), 3)
cv.imwrite('./data1/output.png', img_rgb)
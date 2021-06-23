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
def detect_by_mask(image):
    img = np.copy(image)
    # img_hsv     = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    lower_red = np.array([5, 5, 5])
    upper_red = np.array([210, 210, 200])

    mask = cv.inRange(img_hsv, lower_red, upper_red)
    result = cv.bitwise_and(img, img, mask=mask)
    # cv.imshow('mask', mask)
    # cv.imshow('res', result)
    return result


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
# path = 'cell_counter-Develope\crop.jpg'
path = 'inp.jpg'
img_rgb = cv.imread(path)
# img_rgb = cv.resize(img_rgb, (2200, 1652), interpolation=cv.INTER_AREA)
# cv.imshow('image orginal', img_rgb)


# ------------------RGB-TO-GRAY_----------------------------------
img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
# cv.imshow('cc1', img_gray)


# -----------------GAUSSIAN-FILTER--------------------------------
img_gauss_filter = cv.GaussianBlur(img_gray, (5, 5), 0)
# cv.imshow('cc2', img_gauss_filter)
# histogram_gray(img_gauss_filter)


# -----------------OTSU-THRESHOLDING------------------------------
blur = np.copy(img_gauss_filter)
ret, img_otsu = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cv.imshow('cc3', img_otsu)
cv.imwrite('otsu.png', img_otsu)

# -----------------BINARY-THRESHOLDING----------------------------
# ret,thres = cv.threshold(img_gauss_filter,193,255,cv.THRESH_BINARY)
# cv.imshow('cc4', thres)

# -----------------SOBEL-EDGE-DETECTION---------------------------
# img_edge_detection = detect_edge(img_otsu)
# cv.imshow('edge detection', img_edge_detection)

# -----------------HOUGH-CIRCLE-TRANSFORM---------------------------
gray = np.copy(img_otsu)
rows = gray.shape[0]
# circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 45, param1=200, param2=1, minRadius=1, maxRadius=5)
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 98 , param1=500, param2=6, minRadius=11, maxRadius=19)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        cv.circle(img_rgb, center, 1, (0, 100, 100), 3)
        radius = i[2]
        # cv.circle(img_rgb, center, radius, (255, 0, 255), 3)

# cv.imshow('vcl', img_rgb)
cv.imwrite('output.png', img_rgb)

cv.waitKey(0)
cv.destroyAllWindows()
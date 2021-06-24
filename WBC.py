import os
import time
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img 

# -------------------CONVERT-TO-MASK---------------------------------
def convert_to_mask(path_img, path_msk, path_renew_img): 
    img     = cv.imread(path_img) 
    img_rgb = np.empty((img.shape[0] + 224, img.shape[1] + 224, 3), dtype = int)    
    img_rgb.fill(255)

    for i in range(0, img.shape[0]): 
        for j in range(0, img.shape[1]): 
            img_rgb[i + 112][j + 112] = img[i][j]

    cv.imwrite(path_renew_img, img_rgb)
    img_rgb     = cv.imread(path_renew_img)
    img_gray    = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY) 
    ret, img_thres= cv.threshold(img_gray, 160, 255,cv.THRESH_BINARY_INV)
    cv.imwrite(path_msk, img_thres)

# -------------------CROP-WBC---------------------------------------
def crop_object(path_img, path_msk, path_par): 
    result      = []
    mask        = cv.imread(path_msk)
    img_rgb     = cv.imread(path_img)
    img_gray    = cv.cvtColor(mask, cv.COLOR_RGB2GRAY) 
    img_canny   = cv.Canny(mask, 255, 255)
     
    circles = cv.HoughCircles(img_canny, cv.HOUGH_GRADIENT, 1, img_canny.shape[0] / 20 , param1=200, param2=15, minRadius=20, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1]) 
            crop = np.copy(img_rgb[(i[1]-112):(i[1]+112), (i[0]-112):(i[0]+112)])
            result.append(crop) 
            cv.circle(img_rgb, center, i[2], (0, 100, 100), 2)
     
    cv.imwrite(path_par + 'detection_wbc.png', img_rgb)
    list_img = [] 
    for i in range(0, len(result)): 
        cv.imwrite(path_par + 'wbc' + str(i + 1) + '.png', result[i])
        list_img.append(path_par + 'wbc' + str(i + 1) + '.png') 
    return list_img

# -------------------COUNT-WBC---------------------------------------
def count_wbc(path_par, path_model , path_img): 
    model = load_model(path_model)
    convert_to_mask(path_img, path_par + 'mask.png', path_par + 'img.png')
    list_img = crop_object(path_par + 'img.png', path_par + 'mask.png', path_par) 
    class_wbc = [0, 0, 0, 0, 0] 
    for i in list_img: 
        img = load_img(i, target_size=(224, 224))
        img = img_to_array(img)
        img = np.array([img])
        class_wbc[np.argmax(model.predict)] += 1
    return class_wbc
 
from WBC import count_wbc 
from RBC import count_rbc 

import cv2 as cv
path_par = './test_model/'
path_mod = 'vgg.h5'
path_img = path_par + 'img.png' 
path_msk = path_par + 'mask.png' 
path_dct = path_par + 'detect_rbc.png' 
path_rbc = path_par + 'rbc.png'

num_wbc = count_wbc(path_par, path_mod, path_img='./test_model/7.bmp')
num_rbc = count_rbc(path_par, path_img, path_msk, path_dct, path_rbc)

print('The number of RBCs is:', num_rbc)
print('The number of WBCs is:', sum(num_wbc))
print('The number of Basophil:', num_wbc[0])
print('The number of Eosinophil:', num_wbc[1])
print('The number of Lymphocyte:', num_wbc[2])
print('The number of Monocyte:', num_wbc[3])
print('The number of Neutrophil:', num_wbc[4])
from WBC import count_wbc 
from RBC import count_rbc 

path_par = './test_model/'
path_mod = 'vgg.h5'
path_img = path_par + 'img.png' 
path_msk = path_par + 'mask.png' 
path_dct = path_par + 'detect_rbc.png' 
path_rbc = path_par + 'rbc.png'

num_wbc = count_wbc(path_par, path_mod, path_img='./test_model/7.bmp')
num_rbc = count_rbc(path_par, path_img, path_msk, path_dct, path_rbc)

print(num_rbc)
print(num_wbc)
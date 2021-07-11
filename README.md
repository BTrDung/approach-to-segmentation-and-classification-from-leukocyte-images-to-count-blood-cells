# approach to segmentation and classification from leukocyte images to count blood cells

## Dataset 

The [LISC - *Leukocyte Images for Segmentation and Classification*](http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm) has been used for automatic identification and counting of blood cells

Samples were taken from peripheral blood of 8 normal subjects and 400 samples were obtained from 100 microscope slides. The microscope slides were smeared and stained by Gismo-Right technique and images were acquired by a light microscope (Microscope-Axioskope 40) from the stained peripheral blood using an achromatic lens with a magnification of 100. Then, these images were recorded by a digital camera (Sony Model No. SSCDC50AP) and were saved in the BMP format. The images contain 720×576 pixels.
All of them are color images and were collected from Hematology-Oncology and BMT Research Center of Imam Khomeini hospital in Tehran, Iran. The images were classified by a hematologist into normal leukocytes: **basophil**, **eosinophil**, **lymphocyte**, **monocyte**, and **neutrophil**. Also, the areas related to the nucleus and cytoplasm were manually segmented by an expert.

<p align="center">
  <img src="https://github.com/BTrDung/Complex/blob/master/CreProjCBC/4.bmp" width="700">
</p>

## Requirements
Python: [![Download](https://img.shields.io/badge/download-3.8.11-blue.svg?longCache=true&style=flat&logo=python)](https://www.python.org/downloads/release/python-3811/) 
 
Tensorflow: [![Download](https://img.shields.io/badge/download-2.4.1-blue.svg?longCache=true&style=flat&logo=tensorflow)](https://www.tensorflow.org/) 

OpenCV: [![Download](https://img.shields.io/badge/download-4.5.2.54-blue.svg?longCache=true&style=flat&logo=opencv)](https://opencv.org/) 

Weights: [![Download](https://img.shields.io/badge/download-vgg16.h5-blue.svg?longCache=true&style=flat&logo=google-drive)](https://drive.google.com/drive/folders/13CAH4i3mEc0Ybk14_UJFEJsg_1NTcJIT?usp=sharing) 

## Setup
> pip install numpy 
> 
> pip install tensorflow.keras
> 
> pip install cv2
> 
> pip install matplotlib

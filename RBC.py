import numpy as np
import cv2 
import matplotlib.pyplot as plt

path    = 'E:\Github\Machine-learning-for-counting-blood-cells\cell_counter-Develope\Database\RBC\RBC 28-12 part 1\RBC 17-02\IMG_20200318_171324.jpg'
img = cv2.imread(path) 

#parameter
threshold = 120
low = 10
up = 25

#converting to grayscale
pic1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#croping image
pic1 = pic1[1450:3100,730:2800,]
img = img[1450:3100,730:2800,:]
m = np.copy(img)

#local histogram equalizer
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(pic1)

#threshold
ret,thresh1 = cv2.threshold(cl1,threshold,255,cv2.THRESH_BINARY)

# noise removal
kernel = np.ones((3,3),np.uint8)

sure_bg = cv2.dilate(thresh1,kernel,iterations=3)

unknown = cv2.subtract(sure_bg,thresh1)

# Marker labelling
ret, markers = cv2.connectedComponents(thresh1)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

#markers = watershed(pic, markers, mask=thresh)
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

labels = markers
r = np.zeros(np.max(labels)+1)
i = 0
imgc = np.copy(m)
imgr = np.copy(m)

for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(pic1.shape, dtype="uint8")
    mask[labels == label] = 255
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # draw a circle enclosing the object
    ((x, y), r[i]) = cv2.minEnclosingCircle(c)

    if (r[i]>low)&(r[i]<up):
        cv2.circle(imgr, (int(x), int(y)), int(r[i]), (255,0, 0), 2)
        i +=1
        #cv2.putText(imgc, "#{}".format(i), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255 , 0, 0), 2)
r = r[r!=0]
size = np.size(r)
cell1.append([file,size])
cell2.append(list(r))
cv2.imwrite(os.path.abspath("G:/images/" + file),imgr)
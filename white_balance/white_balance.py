import os
import cv2
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image


filename = 'IMG_9254.JPG'
img = image.load_img(filename)
img_array = np.array(image.img_to_array(img),dtype=np.uint8)
img_shape=img_array.shape
plt.figure()
plt.imshow(img_array)

img_array2 = img_array
img_array2 = np.array(img_array2, dtype=np.uint16)
img_array2[:,:,0] = img_array2[:,:,0] + 52 # white balancing
img_array2 = np.clip(img_array2, 0, 255)
plt.figure()
plt.imshow(img_array2)
cv2.imwrite('wb_'+filename,img_array2[:,:,[2,1,0]])

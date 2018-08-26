# An example of image augmentation 
import numpy as np
import os
from glob import glob
import sys
import random

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from augmentation import augment_img_reshape

im_width = 128
im_height = 128
im_chan = 3

X_train = np.zeros((im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((im_height, im_width, 1), dtype=np.bool_)

# read in the image/mask
img = imread('0a1742c740_image.png')
x_img = resize(img, (im_width, im_height, im_chan), mode='constant', preserve_range=True)
X_train = x_img/255

mask = imread('0a1742c740_mask.png')
Y_train = resize(mask, (im_width, im_height, 1), mode='constant', preserve_range=True)
Y_train = Y_train/65535

# Display image/mask before augmentation
plt.figure()
plt.subplot(1,2,1);imshow(X_train)
plt.subplot(1,2,2);imshow(Y_train.reshape(im_width, im_height))
plt.title('Image/mask before augmentation')

# Reshape to channel-first format preferred by deep learning
X_train_shaped = X_train.reshape(im_chan, im_width, im_height)
Y_train_shaped = Y_train.reshape(1, im_width, im_height)
X_train_shaped = X_train_shaped.astype(np.float32)
Y_train_shaped = Y_train_shaped.astype(np.float32)

# apply image augmentation
aug_image, aug_mask = augment_img_reshape(X_train_shaped, Y_train_shaped)

# Display image/mask after augmentation
plt.figure()
plt.subplot(1,2,1);imshow(aug_image.reshape(im_width, im_height, im_chan))
plt.subplot(1,2,2);imshow(aug_mask.reshape(im_width, im_height))
plt.title('Image/mask after augmentation')
plt.show()
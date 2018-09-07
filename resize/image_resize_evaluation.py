# To evaluate whether image resize is reversible
import numpy as np

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

im_width = 128
im_height = 128

# read in the image/mask
img = imread('0a1742c740_image.png')
img = img/255
img_dim = len(img)

mask = imread('0a1742c740_mask.png')
mask = mask/65535

myOrder = 3;

img2=resize(img,(im_width,im_height), order=myOrder)
img3=resize(img2,(img_dim,img_dim), order=myOrder)
img_delta = np.sum(np.abs(img-img3)**2)
print(img_delta)

mask2=resize(mask,(im_width,im_height), order=myOrder)
mask3=resize(mask2,(img_dim,img_dim), order=myOrder)
mask_delta = np.sum(np.abs(mask-mask3)**2)
print(mask_delta)

plt.figure()
plt.subplot(1,2,1);imshow(img);plt.title('Original Image');
plt.subplot(1,2,2);imshow(mask);plt.title('Original Mask');

plt.figure()
plt.subplot(1,2,1);imshow(img[:,:,0]-img3[:,:,0]);plt.title('Delta Image');
plt.subplot(1,2,2);imshow(mask-mask3);plt.title('Delta Mask');
plt.show()


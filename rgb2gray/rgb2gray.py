import cv2
import numpy as np
import matplotlib.pyplot as plt

img_name = 'lena_color_512.tif';
img_color = cv2.imread(img_name, cv2.IMREAD_COLOR)
img_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

shape = img_gray.shape
img_color2gray = np.zeros(shape, dtype=np.int32)
img_color2gray[:,:] = np.round(0.299*img_color[:,:,2]+0.587*img_color[:,:,1]+0.114*img_color[:,:,0])
img_color2gray = np.clip(img_color2gray, 0, 255)

print('Max diff between img_gray and img_color2gray is '+str(np.max(np.abs(img_gray-img_color2gray))))
print('Min diff between img_gray and img_color2gray is '+str(np.min(np.abs(img_gray-img_color2gray))))
print('Total diff between img_gray and img_color2gray is '+str(np.sum(np.abs(img_gray-img_color2gray))))

plt.imshow(np.abs(img_gray-img_color2gray))
plt.show()

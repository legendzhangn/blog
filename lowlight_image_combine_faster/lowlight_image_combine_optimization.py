import numpy as np
#import jax.numpy as jnp
from matplotlib import pylab as plt
import rawpy
import cv2
import time
from multiprocessing import Pool

def image_align(rgb_raw_image_group):
    search_range = 40;
    block_size = 1024;
    rgb_raw_image = rgb_raw_image_group[0]
    rgb_raw_image_candi = rgb_raw_image_group[1]
    [height, width] = rgb_raw_image.shape;
    target_total = np.zeros((height, width), dtype=np.uint16)
    target_cnt = np.zeros((height, width), dtype=np.uint16)
    target_group = np.zeros((2,height, width), dtype=np.uint16)

    # Calculate index of each block
    row_index = list(range(0,height,block_size))
    if (row_index[len(row_index)-1] != height):
        row_index.append(height)
    col_index = list(range(0,width,block_size))
    if (col_index[len(col_index)-1] != width):
        col_index.append(width)
    if ((row_index[len(row_index)-1] - row_index[len(row_index)-2] < search_range) | (col_index[len(col_index)-1] - col_index[len(col_index)-2] < search_range)):
        print("The size of last block is smaller than search range!")

    search_metric = np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32)
    boundary_adjust = np.zeros((2*search_range+1, 2*search_range+1, 4), dtype=np.int16) # for each search index, [row_start, row_end, col_start, col_end]
    target_total[:,:] = 0
    target_cnt[:,:] = 0
    #profile_sum=0
    for rowblk in range(len(row_index)-1):
        for colblk in range(len(col_index)-1):
            row_start = row_index[rowblk]; row_end = row_index[rowblk+1];
            col_start = col_index[colblk]; col_end = col_index[colblk+1];
            search_metric[:,:] = 0
            boundary_adjust[:,:,:] = 0
            for row in range(2*search_range+1):
                for col in range(2*search_range+1):
                    row_offset = row - search_range
                    col_offset = col - search_range
                    if ((row_offset % 2 == 0) & (col_offset % 2 == 0)):
                        # Boundary detection
                        if (row_start+row_offset < 0):
                            boundary_adjust[row, col, 0] = - (row_start+row_offset)
                        if (row_end+row_offset > height):
                            boundary_adjust[row, col, 1] = height - (row_end+row_offset)
                        if (col_start+col_offset < 0):
                            boundary_adjust[row, col, 2] = - (col_start+col_offset)
                        if (col_end+col_offset > width):
                            boundary_adjust[row, col, 3] = width - (col_end+col_offset)
                        candidate = rgb_raw_image_candi[row_start+row_offset+boundary_adjust[row, col, 0]:row_end+row_offset+boundary_adjust[row, col, 1]:2,
                        col_start+col_offset+boundary_adjust[row, col, 2]:col_end+col_offset+boundary_adjust[row, col, 3]:2]
                        target = rgb_raw_image[row_start+boundary_adjust[row, col, 0]:row_end+boundary_adjust[row, col, 1]:2,
                        col_start+boundary_adjust[row, col, 2]:col_end+boundary_adjust[row, col, 3]:2]
                        # Calculate search metric
                        search_metric[row,col] = np.mean(np.abs(target.astype(np.int32)-candidate.astype(np.int32)))
                    else:
                        search_metric[row,col] = 65535
            row_offset_final = np.where(search_metric == search_metric.min())[0][0] - search_range
            col_offset_final = np.where(search_metric == search_metric.min())[1][0] - search_range
            boundary_final = boundary_adjust[row_offset_final+search_range,col_offset_final+search_range,:]

            target_total[row_start+boundary_final[0]:row_end+boundary_final[1]
            ,col_start+boundary_final[2]:col_end+boundary_final[3]] += rgb_raw_image_candi[
            row_start+row_offset_final+boundary_final[0]:row_end+row_offset_final+boundary_final[1]
            ,col_start+col_offset_final+boundary_final[2]:col_end+col_offset_final+boundary_final[3]]
            target_cnt[row_start+boundary_final[0]:row_end+boundary_final[1]
            ,col_start+boundary_final[2]:col_end+boundary_final[3]] += 1

    target_group[0] = target_total
    target_group[1] = target_cnt
    return target_group

raw_py = rawpy.imread('images/20200328_105350.dng') # load image

no_auto_bright = 1;
t_white = 0
[height, width] = raw_py.raw_image.shape;

candi_files = ['images/20200328_105351.dng','images/20200328_105353.dng','images/20200328_105355.dng']


wb = raw_py.camera_whitebalance
color_matrix = raw_py.color_matrix
rgb = raw_py.postprocess(use_camera_wb=True,half_size=False)
rgb_raw_image = raw_py.raw_image
plt.figure();plt.imshow(rgb);plt.title('Reference')
pic = np.zeros((width, height, 3), dtype=np.uint16)
pic_temp = np.zeros((width, height, 3), dtype=np.float32)
rgb_raw_image_candi = np.zeros((len(candi_files), height, width), dtype=np.uint16)
target_total = np.zeros((height, width), dtype=np.uint32)
target_cnt = np.zeros((height, width), dtype=np.uint16)

for f in range(len(candi_files)):
    raw_py1 = rawpy.imread(candi_files[f])
    rgb_raw_image_candi[f] = raw_py1.raw_image


# Single thread processing
start = time.time()

rgb_raw_image_group = np.zeros((2,height, width), dtype=np.uint16)
target_group = np.zeros((2,height, width), dtype=np.uint16)
target_total[:,:] = rgb_raw_image.astype(np.uint32)
target_cnt[:,:] = 1
for f in range(len(candi_files)):
    rgb_raw_image_group[0] = rgb_raw_image
    rgb_raw_image_group[1] = rgb_raw_image_candi[f]
    target_group = image_align(rgb_raw_image_group)
    target_total += target_group[0].astype(np.uint32)
    target_cnt += target_group[1]


end = time.time()
print('Time spent by using decimated image for alignment is: '+str(end - start)+' s')


# Multi thread processing
start = time.time()
rgb_raw_image_group = np.zeros((2,height, width), dtype=np.uint16)
target_group = np.zeros((2,height, width), dtype=np.uint16)
target_total[:,:] = rgb_raw_image.astype(np.uint32)
target_cnt[:,:] = 1
image_input = []
# Init the list first. image_input.append() has pointer problem
for f in range(len(candi_files)):
    image_input.append(np.zeros((2,height, width), dtype=np.uint16))
for f in range(len(candi_files)):
    image_input[f][0,:,:] = rgb_raw_image[:,:]
    image_input[f][1,:,:] = rgb_raw_image_candi[f,:,:]

with Pool(6) as p:
    image_output = p.map(image_align, image_input)

for f in range(len(candi_files)):
    target_group = image_output[f]
    target_total += target_group[0].astype(np.uint32)
    target_cnt += target_group[1]

end = time.time()
print('Time spent by using decimated image for alignment + multi-core processing is: '+str(end - start)+' s')


# Raw image combining
target_total = np.clip(np.divide(target_total,target_cnt),0,65535)
rgb_raw_image[:,:] = target_total.astype(np.uint16) # Modify content not pointer

# Post processing of raw image
rgb2 = raw_py.postprocess(use_camera_wb=True,half_size=False)
plt.figure();plt.imshow(rgb2);plt.title('Reconstructed')

bilateral = cv2.bilateralFilter(rgb, 10, 75, 75)
plt.figure();plt.imshow(bilateral);plt.title('Bilateral Original 10, 75, 75')

bilateral2 = cv2.bilateralFilter(rgb2, 10, 75, 75)
plt.figure();plt.imshow(bilateral2);plt.title('Bilateral Reconstructed 10, 75, 75')

plt.show()
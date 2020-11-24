import numpy as np
#import jax.numpy as jnp
from matplotlib import pylab as plt
import rawpy
import cv2
import time

start = time.time()


raw_py = rawpy.imread('images/20200411_105233.dng') # Galaxy S9 pro


no_auto_bright = 1;
t_white = 0
[height, width] = raw_py.raw_image.shape;

search_range = 40;
block_size = 1024;


# Galaxy S9 pro
candi_files = ['images/20200411_105235.dng','images/20200411_105237.dng','images/20200411_105238.dng','images/20200411_105240.dng',
               'images/20200411_105242.dng','images/20200411_105244.dng','images/20200411_105246.dng']


wb = raw_py.camera_whitebalance
color_matrix = raw_py.color_matrix
rgb = raw_py.postprocess(use_camera_wb=True,half_size=False)
rgb_raw_image = raw_py.raw_image
#rgb = raw_py.postprocess(half_size=True)
plt.figure();plt.imshow(rgb);plt.title('Reference')
#rgb_raw_image[1000:2000,1000:2000] = 0;
#rgb_hole = raw_py.postprocess(use_camera_wb=True,half_size=False)
#plt.figure();plt.imshow(rgb_hole);plt.title('Hole')
pic = np.zeros((width, height, 3), dtype=np.uint16)
pic_temp = np.zeros((width, height, 3), dtype=np.float32)
rgb_raw_image_candi = np.zeros((len(candi_files), height, width), dtype=np.uint16)
target_total = np.zeros((height, width), dtype=np.uint32)
target_cnt = np.zeros((height, width), dtype=np.uint16)

# Calculate index of each block
row_index = list(range(0,height,block_size))
if (row_index[len(row_index)-1] != height):
    row_index.append(height)
col_index = list(range(0,width,block_size))
if (col_index[len(col_index)-1] != width):
    col_index.append(width)
if ((row_index[len(row_index)-1] - row_index[len(row_index)-2] < search_range) | (col_index[len(col_index)-1] - col_index[len(col_index)-2] < search_range)):
    print("The size of last block is smaller than search range!")

for f in range(len(candi_files)):
    raw_py1 = rawpy.imread(candi_files[f])
    rgb_raw_image_candi[f] = raw_py1.raw_image


# Decimation
rgb_raw_image_deci8 = np.zeros((height//8, width//8), dtype=np.uint16)
rgb_raw_image_candi_deci8 = np.zeros((len(candi_files), height//8, width//8), dtype=np.uint16)
deci4_offset = 2
kernel = np.array([[2, 4, 5, 4, 2],
                [4, 9, 12, 9, 4],
                [5, 12, 15, 12, 5],
                [4, 9, 12, 9, 4],
                [2, 4, 5, 4, 2]])/159
rgb_raw_image_deci8[:,:] = cv2.filter2D(rgb_raw_image[::2,::2],-1,kernel)[deci4_offset::4,deci4_offset::4]
for f in range(len(candi_files)):
    rgb_raw_image_candi_deci8[f, :,:] = cv2.filter2D(rgb_raw_image_candi[f,::2,::2],-1,kernel)[deci4_offset::4,deci4_offset::4]


search_range = 4
search_metric_save = []
search_metric_sum_save = []
# Init the list first. image_input.append() has pointer problem
for f in range(len(candi_files)*(len(row_index)-1)*(len(col_index)-1)):
    search_metric_save.append(np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32))
    search_metric_sum_save.append(np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32))

# Image alignment based on original image decimated by 8
search_metric = np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32)
search_metric_sum = np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32)
boundary_adjust = np.zeros((2*search_range+1, 2*search_range+1, 4), dtype=np.int16) # for each search index, [row_start, row_end, col_start, col_end]
offset_final = np.zeros((len(candi_files),len(row_index)-1,len(col_index)-1, 2), dtype=np.int16)
search_metric_cnt = 0
#profile_sum=0
for f in range(len(candi_files)):
    for rowblk in range(len(row_index)-1):
        for colblk in range(len(col_index)-1):
            row_start = row_index[rowblk]//8; row_end = row_index[rowblk+1]//8;
            col_start = col_index[colblk]//8; col_end = col_index[colblk+1]//8;
            search_metric[:,:] = 0
            search_metric_sum[:,:] = 0
            boundary_adjust[:,:,:] = 0
            for row in range(2*search_range+1):
                for col in range(2*search_range+1):
                    row_offset = row - search_range
                    col_offset = col - search_range
                    if ((row_offset % 2 == 0) & (col_offset % 2 == 0)):
                        # Boundary detection
                        if (row_start+row_offset < 0):
                            boundary_adjust[row, col, 0] = - (row_start+row_offset)
                        if (row_end+row_offset > height//8):
                            boundary_adjust[row, col, 1] = height//8 - (row_end+row_offset)
                        if (col_start+col_offset < 0):
                            boundary_adjust[row, col, 2] = - (col_start+col_offset)
                        if (col_end+col_offset > width//8):
                            boundary_adjust[row, col, 3] = width//8 - (col_end+col_offset)
                        #print(boundary_adjust[row, col, :])
                        candidate = rgb_raw_image_candi_deci8[f,row_start+row_offset+boundary_adjust[row, col, 0]:row_end+row_offset+boundary_adjust[row, col, 1],
                        col_start+col_offset+boundary_adjust[row, col, 2]:col_end+col_offset+boundary_adjust[row, col, 3]]
                        target = rgb_raw_image_deci8[row_start+boundary_adjust[row, col, 0]:row_end+boundary_adjust[row, col, 1],
                        col_start+boundary_adjust[row, col, 2]:col_end+boundary_adjust[row, col, 3]]
                        # Calculate search metric
                        search_metric[row,col] = np.mean(np.abs(target.astype(np.int32)-candidate.astype(np.int32)))

                    else:
                        search_metric[row,col] = 65535
            row_offset_final = np.where(search_metric == search_metric.min())[0][0] - search_range
            col_offset_final = np.where(search_metric == search_metric.min())[1][0] - search_range
            boundary_final = boundary_adjust[row_offset_final+search_range,col_offset_final+search_range,:]

            offset_final[f,rowblk,colblk] = [row_offset_final, col_offset_final] # save offset final

            search_metric_save[search_metric_cnt][:,:] = search_metric[:,:]
            search_metric_sum_save[search_metric_cnt][:,:] = search_metric_sum[:,:]
            search_metric_cnt = search_metric_cnt + 1



# Image alignment based on original image decimated by 2
search_range = 8 # search [-4,4] after by-2 decimation
search_metric_save = []
search_metric_sum_save = []
# Init the list first. image_input.append() has pointer problem
for f in range(len(candi_files)*(len(row_index)-1)*(len(col_index)-1)):
    search_metric_save.append(np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32))
    search_metric_sum_save.append(np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32))

# Raw image alignment
search_metric = np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32)
search_metric_sum = np.zeros((2*search_range+1, 2*search_range+1), dtype=np.float32)
boundary_adjust = np.zeros((2*search_range+1, 2*search_range+1, 4), dtype=np.int16) # for each search index, [row_start, row_end, col_start, col_end]
target_total[:,:] = rgb_raw_image
target_cnt[:,:] = 1
search_metric_cnt = 0
#profile_sum=0
for f in range(len(candi_files)):
    for rowblk in range(len(row_index)-1):
        for colblk in range(len(col_index)-1):
            row_start = row_index[rowblk]; row_end = row_index[rowblk+1];
            col_start = col_index[colblk]; col_end = col_index[colblk+1];
            search_metric[:,:] = 0
            search_metric_sum[:,:] = 0
            boundary_adjust[:,:,:] = 0
            for row in range(2*search_range+1):
                for col in range(2*search_range+1):
                    row_offset = row - search_range + (offset_final[f,rowblk,colblk,0]*4+2)*2
                    col_offset = col - search_range + (offset_final[f,rowblk,colblk,1]*4+2)*2
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
                        #print(boundary_adjust[row, col, :])
                        candidate = rgb_raw_image_candi[f,row_start+row_offset+boundary_adjust[row, col, 0]:row_end+row_offset+boundary_adjust[row, col, 1]:2,
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
            row_offset_final += (offset_final[f,rowblk,colblk,0]*4+2)*2 # change here so that the index of boundary_final is still correct
            col_offset_final += (offset_final[f,rowblk,colblk,1]*4+2)*2

            target_total[row_start+boundary_final[0]:row_end+boundary_final[1]
            ,col_start+boundary_final[2]:col_end+boundary_final[3]] += rgb_raw_image_candi[f,
            row_start+row_offset_final+boundary_final[0]:row_end+row_offset_final+boundary_final[1]
            ,col_start+col_offset_final+boundary_final[2]:col_end+col_offset_final+boundary_final[3]]
            target_cnt[row_start+boundary_final[0]:row_end+boundary_final[1]
            ,col_start+boundary_final[2]:col_end+boundary_final[3]] += 1

            search_metric_save[search_metric_cnt][:,:] = search_metric[:,:]
            search_metric_sum_save[search_metric_cnt][:,:] = search_metric_sum[:,:]
            search_metric_cnt = search_metric_cnt + 1


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

end = time.time()
print('Time spent is: '+str(end - start)+' s')

plt.show()

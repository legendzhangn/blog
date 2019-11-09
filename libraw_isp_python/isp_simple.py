import numpy as np
from matplotlib import pylab as plt
import rawpy
#plt.rcParams['interactive'] = True
raw_py = rawpy.imread('images/20191109_073417.dng')

#wb = raw_py.daylight_whitebalance
wb = raw_py.camera_whitebalance
color_matrix = raw_py.color_matrix
rgb = raw_py.postprocess(use_camera_wb=True,half_size=True)
#rgb = raw_py.postprocess(half_size=True)
plt.figure();plt.imshow(rgb);plt.title('Reference')
pic = np.zeros((980, 2016, 3), dtype=np.uint16)
pic_temp = np.zeros((980, 2016, 3), dtype=np.float32)
color4 = np.zeros((980, 2016, 4), dtype=np.float32)
color4_noscale = np.zeros((980, 2016, 4), dtype=np.float32)

# Scale white balance coefficients
wb[3] = wb[1];
wb_scale = [0]*len(wb)
for i in range(len(wb)):
    wb_scale[i] = wb[i]/min(wb)*65535/1023;

# white balancing
color4_noscale[:,:,0] = raw_py.raw_image[0::2,1::2]
color4_noscale[:,:,1] = raw_py.raw_image[0::2,0::2]
color4_noscale[:,:,2] = raw_py.raw_image[1::2,0::2]
color4_noscale[:,:,3] = raw_py.raw_image[1::2,1::2]

color4[:,:,0] = raw_py.raw_image[0::2,1::2]*wb_scale[0]
color4[:,:,1] = raw_py.raw_image[0::2,0::2]*wb_scale[1]
color4[:,:,2] = raw_py.raw_image[1::2,0::2]*wb_scale[2]
color4[:,:,3] = raw_py.raw_image[1::2,1::2]*wb_scale[3]

# mix green
color4[:,:,1] = (color4[:,:,1] + color4[:,:,3])/2

# clipping to [0, 65535]
color4 = np.clip(color4,0,65535)



# Generate Gamma curve
g = [0.450000, 4.500000, 0.081243, 0.018054, 0.099297, 0.517181];
gamma_curve = np.zeros((65536), dtype=np.uint16);
for i in range(65536):
    if (i/65535 < g[3]):
        gamma_curve[i] = int(i*g[1])
    else:
        gamma_curve[i] = int(65535*(np.power(1.0*i/65535, g[0])*(1+g[4])-g[4]))

# Apply color matrix

pic_temp[:,:,0] = color_matrix[0,0]*color4[:,:,0] + color_matrix[0,1] * color4[:,:,1] + color_matrix[0,2] * color4[:,:,2]
pic_temp[:,:,1] = color_matrix[1,0]*color4[:,:,0] + color_matrix[1,1] * color4[:,:,1] + color_matrix[1,2] * color4[:,:,2]
pic_temp[:,:,2] = color_matrix[2,0]*color4[:,:,0] + color_matrix[2,1] * color4[:,:,1] + color_matrix[2,2] * color4[:,:,2]

pic_temp = np.clip(pic_temp,0,65535)
pic_temp = pic_temp.astype(np.uint16)
pic_temp = gamma_curve[pic_temp]

pic_temp = np.clip(1.0*pic_temp/65535*255,0,255)
pic=pic_temp.astype(np.uint8)
pic = np.rot90(pic, k=3);
plt.figure();plt.imshow(pic);plt.title('Reconstructed')

plt.figure();plt.imshow(np.abs(pic.astype(np.int16)-rgb.astype(np.int16)));plt.title('Delta')
plt.show();
print("max error = "+ str(np.max(np.abs(pic.astype(np.int16)-rgb.astype(np.int16)))))

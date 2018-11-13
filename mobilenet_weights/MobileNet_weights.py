import numpy as np
import matplotlib.pyplot as plt
import keras

base_model = keras.applications.mobilenet.MobileNet(weights='imagenet');

base_model.summary()

W = base_model.get_weights();
epsilon = 0.001

print(len(W))

for i in range(len(W)):
  print(W[i].shape)
  
# To plot the weights of used in convolution and FC
plt.figure()
plt.subplot(3,3,1);plt.hist(W[0].flatten());plt.title('conv1');
plt.subplot(3,3,2);plt.hist(W[5].flatten());plt.title('conv_dw_1');
plt.subplot(3,3,3);plt.hist(W[10].flatten());plt.title('conv_pw_1');
plt.subplot(3,3,4);plt.hist(W[35].flatten());plt.title('conv_dw_4');
plt.subplot(3,3,5);plt.hist(W[40].flatten());plt.title('conv_pw_4');
plt.subplot(3,3,6);plt.hist(W[65].flatten());plt.title('conv_dw_7');
plt.subplot(3,3,7);plt.hist(W[125].flatten());plt.title('conv_dw_13');
plt.subplot(3,3,8);plt.hist(W[130].flatten());plt.title('conv_pw_13');
plt.subplot(3,3,9);plt.hist(W[135].flatten());plt.title('conv_preds');


# To plot the distribution of scale used in batch normalization
plt.figure();
ln=0;
plt.subplot(3,3,1);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv1_bn');
ln=5;
plt.subplot(3,3,2);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_dw_1_bn');
ln=10;
plt.subplot(3,3,3);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_pw_1_bn');
ln=45;
plt.subplot(3,3,4);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_dw_5_bn');
ln=50;
plt.subplot(3,3,5);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_pw_5_bn');
ln=85;
plt.subplot(3,3,6);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_dw_9_bn');
ln=90;
plt.subplot(3,3,7);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_pw_9_bn');
ln=125;
plt.subplot(3,3,8);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_dw_13_bn');
ln=130
plt.subplot(3,3,9);plt.hist((W[ln+1]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_pw_13_bn');


# To plot the distribution of bias used in batch normalization
plt.figure();
ln=0
plt.subplot(3,3,1);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv1_bn');
ln=5
plt.subplot(3,3,2);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_dw_1_bn');
ln=10
plt.subplot(3,3,3);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_pw_1_bn');
ln=45
plt.subplot(3,3,4);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_dw_5_bn');
ln=50
plt.subplot(3,3,5);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_pw_5_bn');
ln=85
plt.subplot(3,3,6);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_dw_9_bn');
ln=90
plt.subplot(3,3,7);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_pw_9_bn');
ln=125
plt.subplot(3,3,8);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_dw_13_bn');
ln=130
plt.subplot(3,3,9);plt.hist((W[ln+2]-W[ln+1]*W[ln+3]/np.sqrt(W[ln+4]+epsilon)).flatten());plt.title('conv_pw_13_bn');
plt.show()

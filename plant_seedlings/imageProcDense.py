# Read image and show it
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from keras.preprocessing import image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


from keras.models import Model
from keras.applications.xception import *
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import densenet
import csv

batch_size = 16
#epochs = 500
epochs = 1
img_rows = 299;
img_cols = img_rows;
imageSize = (img_cols, img_rows);
plant_names =['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',
  'Small-flowered Cranesbill', 'Sugar beet'];
num_plants = len(plant_names);

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)


# Read in test files
init = 0;
testFileList = glob.glob('./test/*');
print(str(len(testFileList))+' test images')
for myFile in range(0, len(testFileList)):
  img = cv2.imread(testFileList[myFile], cv2.IMREAD_COLOR);
  img_norm = cv2.resize(img, imageSize);
  if (init == 0):
    init = 1;
    image_test = np.array([img_norm]);
  else:
    image_test = np.append(image_test, [img_norm], axis=0);

if K.image_data_format() == 'channels_first':
    image_test = image_test.reshape(image_test.shape[0], 3, img_rows, img_cols)
else:
    image_test = image_test.reshape(image_test.shape[0], img_rows, img_cols, 3)

#print(image_test)

image_test = image_test.astype('float32')
image_test /= 255
print('image_test shape:', image_test.shape)
print(image_test.shape[0], 'test samples')

base_model = densenet.DenseNet169(weights='imagenet', input_shape=input_shape, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights("DenseNet169.hdf5")
predict_results = model.predict(image_test);

base_model3 = densenet.DenseNet201(weights='imagenet', input_shape=input_shape, include_top=False)
x3 = base_model3.output
x3 = GlobalAveragePooling2D()(x3)
x3 = Dropout(0.5)(x3)
x3 = Dense(1024, activation='relu')(x3)
x3 = Dropout(0.5)(x3)
predictions3 = Dense(12, activation='softmax')(x3)
model3 = Model(inputs=base_model3.input, outputs=predictions3)

model3.load_weights("DenseNet201.hdf5")
predict_results3 = model3.predict(image_test);

predict_results += predict_results3;

writer = csv.writer(open("predictLog_DenseNet201_plus_DenseNet169_gen.csv", "w"), lineterminator='\n');
writer.writerow(['filename','Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',
  'Small-flowered Cranesbill', 'Sugar beet'])
for i in range(0, len(predict_results)):
  writer.writerow([testFileList[i].split('/')[2]]+predict_results[i].tolist());

writer_sub = csv.writer(open("submissionLog_DenseNet201_plus_DenseNet169_gen.csv", "w"), lineterminator='\n');
writer_sub.writerow(['file', 'species']);
for i in range(0, len(predict_results)):
  writer_sub.writerow([testFileList[i].split('/')[2]]+[plant_names[np.argmax(predict_results[i])]])

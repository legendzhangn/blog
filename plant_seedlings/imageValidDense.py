# Read image and show it
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from keras.preprocessing import image
import keras
from keras.models import Model
from keras.applications.xception import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import densenet
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import csv

batch_size = 8
epochs = 100
img_rows = 299;
img_cols = img_rows;
imageSize = (img_cols, img_rows);
plant_names =['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',
  'Small-flowered Cranesbill', 'Sugar beet'];
num_plants = len(plant_names);

# Read in training files
folder_train = [];
init = 0;
print('Start to read in training files');
for folder in range(0, num_plants):
#for folder in range(0, 2):
  fileList = glob.glob('./train/'+plant_names[folder]+'/*');
  print('Folder '+plant_names[folder]+' has '+str(len(fileList))+' files')
  for myFile in range(0, len(fileList)):
    img = cv2.imread(fileList[myFile], cv2.IMREAD_COLOR);
    img_norm = cv2.resize(img, imageSize);
    folder_train.append(folder);
    if (init == 0):
      init = 1;
      image_train = np.array([img_norm]);
    else:
      image_train = np.append(image_train, [img_norm], axis=0);


print(len(folder_train))
cat_train = keras.utils.to_categorical(folder_train, num_plants)
#print(cat_train)
#print(cat_train[0])


if K.image_data_format() == 'channels_first':
    image_train = image_train.reshape(image_train.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    image_train = image_train.reshape(image_train.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

#print(image_train)


image_train = image_train.astype('float32')
image_train /= 255
#image_train /= 127.5
#image_train -= 1.0
print('image_train shape:', image_train.shape)
print(image_train.shape[0], 'train samples')

# Split train and validation sets
X_train, X_test, y_train, y_test = train_test_split(image_train, cat_train, test_size=0.2, random_state=101, stratify=cat_train)

# Model training starts
base_model = densenet.DenseNet201(weights='imagenet', input_shape=input_shape, include_top=False)
#base_model = densenet.DenseNet169(weights='imagenet', input_shape=input_shape, include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

gen = ImageDataGenerator(
        rotation_range=360.,
		width_shift_range=0.3,
		height_shift_range=0.3,
		zoom_range=0.3,
		horizontal_flip=True,
		vertical_flip=True
)

# checkpoint
#filepath="DenseNet169.hdf5"
filepath="DenseNet201.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(gen.flow(X_train, y_train,batch_size=batch_size),
		   steps_per_epoch=len(image_train)/batch_size,
		   epochs=epochs,
		   verbose=2,
		   callbacks=callbacks_list,
		   validation_data=(X_test, y_test))

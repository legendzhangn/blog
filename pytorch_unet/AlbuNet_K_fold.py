# A Pytorch example of UNet + K fold cross vaildation
# Baseline from Erik Istre's script in https://www.kaggle.com/erikistre/pytorch-basic-u-net
# Add K-fold vaildation, AlbuNet, Stratification, data augmentation, etc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from glob import glob
import sys
import random

from tqdm import tqdm_notebook
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics import jaccard_similarity_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
from unet_models import AlbuNet
import torchvision.transforms as transforms
from augmentation import augment_img_reshape
from transform_ck import train_augment, do_resize2, do_center_pad_to_factor2, compute_center_pad
from evaluate_ck import do_kaggle_metric
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set some parameters# Set s
epoch_num = 500
im_width = 256
im_height = 256
IMAGE_HEIGHT, IMAGE_WIDTH = 202, 202

DY0, DY1, DX0, DX1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=64)


im_chan = 3
path_train = './input/train'
path_test = './input/test'

train_path_images = os.path.abspath(path_train + "/images/")
train_path_masks = os.path.abspath(path_train + "/masks/")

test_path_images = os.path.abspath(path_test + "/images/")
test_path_masks = os.path.abspath(path_test + "/masks/")
train_path_images_list = glob(os.path.join(train_path_images, "*.png"))
train_path_masks_list = glob(os.path.join(train_path_masks, "*.png"))
test_path_images_list = glob(os.path.join(test_path_images, "*.png"))
test_path_masks_list = glob(os.path.join(test_path_masks, "*.png"))

train_ids = next(os.walk(train_path_images))[2]
test_ids = next(os.walk(test_path_images))[2]
# Get and resize train images and masks
X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool_)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()


for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
    img = imread(path_train + '/images/' + id_)
    mask = imread(path_train + '/masks/' + id_)
    img0, mask0 = do_resize2(img, mask, 202, 202)
    img0, mask0 = do_center_pad_to_factor2(img0, mask0, factor=64)
    X_train[n] = img0
    Y_train[n] = mask0.reshape(256, 256, 1)

print('Done!')

# https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader
class saltIDDataset(torch.utils.data.Dataset):

    def __init__(self,preprocessed_images,train=True, transform=None, preprocessed_masks=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.train = train
        self.transform = transform
        self.images = preprocessed_images
        if self.train:
            self.masks = preprocessed_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = None
        if self.train:
            mask = self.masks[idx]

            if self.transform is not None:
                image, mask = self.transform(image, mask)

        return (image, mask)


# Reshape datasets
X_train_shaped = np.rollaxis(X_train,3,1)
X_train_shaped = X_train_shaped/255
Y_train_shaped = Y_train.reshape(-1, 1, im_width, im_height)
X_train_shaped = X_train_shaped.astype(np.float32)
Y_train_shaped = Y_train_shaped.astype(np.float32)


torch.cuda.manual_seed_all(4200)
np.random.seed(133700)


# Stratification: data binning based on salt size in mask. Divide each category to training and validation data
ind = np.arange(len(Y_train_shaped))
np.random.shuffle(ind)
coverage = []
for i in range(0, len(Y_train_shaped)):
  coverage.append(np.sum(Y_train_shaped[ind[i]]))

hist, bin_edges = np.histogram(coverage)
# In np.digitize, each index i returned is such that bins[i-1] <= x < bins[i]
# Need to increase the last bin_edges by 1 to avoid genarating a new category with digitize
bin_edges[len(bin_edges)-1] = bin_edges[len(bin_edges)-1] + 1
cindex = np.digitize(coverage,bin_edges)

val_size = 2/10
for ii in range(5): #5-fold learning
    k = ii
    print('Training for '+str(k)+' of 5 fold starts!')
    train_idxs = []
    val_idxs = []
    for i in range(0,10):
      index_temp = ind[cindex==i+1]
      list_temp = index_temp.T.tolist()
      val_samples = round(len(index_temp)*val_size)
      if (k == 0):
          val_idxs = val_idxs + list_temp[:val_samples]
          train_idxs = train_idxs + list_temp[val_samples:]
      elif (k == 4):
          val_idxs = val_idxs + list_temp[4*val_samples:]
          train_idxs = train_idxs + list_temp[:4*val_samples]
      else:
          val_idxs = val_idxs + list_temp[k*val_samples:(k+1)*val_samples]
          train_idxs = train_idxs + list_temp[:k*val_samples] + list_temp[(k+1)*val_samples:]
    val_idxs = ind[val_idxs]
    train_idxs = ind[train_idxs]
    print('val size')
    print(len(val_idxs))
    print('train size')
    print(len(train_idxs))

    # Load data
    salt_ID_dataset_train = saltIDDataset(X_train_shaped[train_idxs],
                                          train=True,
    									  #transform = augment_img_reshape,
                                          transform = train_augment,
    									  #transform = None,
                                          preprocessed_masks=Y_train_shaped[train_idxs])
    salt_ID_dataset_val = saltIDDataset(X_train_shaped[val_idxs],
                                          train=True,
                                          preprocessed_masks=Y_train_shaped[val_idxs])

    batch_size = 8

    train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_val,
                                               batch_size=batch_size,
                                               shuffle=False)



    model = AlbuNet(pretrained=True, is_deconv=True);
    model.cuda();

    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training started");

    mean_train_losses = []
    mean_val_losses = []
    min_valid_loss = 10000
    max_kaggle_metric = 0
    for epoch in range(epoch_num):
        train_losses = []
        val_losses = []
        predicts = []
        truths   = []
        train_accu = 0;
        valid_accu = 0;
        for images, masks in train_loader:
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())

            outputs = model(images)

            loss = criterion(outputs, masks)
            train_losses.append(loss.data)


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


    		#Calculate binary accuracy
            train_accu = train_accu + torch.sum(torch.round(torch.sigmoid(outputs))==masks);
        for images, masks in val_loader:
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_losses.append(loss.data)

    		#Calculate binary accuracy
            valid_accu = valid_accu + torch.sum(torch.round(torch.sigmoid(outputs))==masks);

            outputs = outputs[:,:,DY0:DY0+IMAGE_HEIGHT, DX0:DX0+IMAGE_WIDTH]
            masks   = masks  [:,:,DY0:DY0+IMAGE_HEIGHT, DX0:DX0+IMAGE_WIDTH]
            outputs = F.avg_pool2d(outputs, kernel_size = 2, stride = 2)
            masks = F.avg_pool2d(masks, kernel_size = 2, stride = 2)
            predicts.append(outputs.data.cpu().numpy())
            truths.append(masks.data.cpu().numpy())


        # Update kaggle metric
        predicts = np.concatenate(predicts).squeeze()
        truths   = np.concatenate(truths).squeeze()
        precision, result, threshold  = do_kaggle_metric(predicts, truths)


        mean_train_losses.append(np.mean(train_losses))
        mean_val_losses.append(np.mean(val_losses))
        train_accu_epoch = train_accu.float()/im_width/im_height/((1-val_size) * len(X_train_shaped))
        valid_accu_epoch = valid_accu.float()/im_width/im_height/(val_size * len(X_train_shaped))
        # Print Loss
        print('Epoch: {}. Train Loss: {}. Val Loss: {}'.format(epoch+1, np.mean(train_losses), np.mean(val_losses)))
        print('Epoch: {}. Train Accuracy: {}. Val Accuracy: {}'.format(epoch+1, train_accu_epoch, valid_accu_epoch))
        print('Epoch: {}. Kaggle metric: {}'.format(epoch+1, precision.mean()))

        if (precision.mean() > max_kaggle_metric):
          max_kaggle_metric = precision.mean();
          torch.save(model.state_dict(), 'basic_unit_pytorch_256_5fold'+str(k)+'_albu.pt')
          print('save basic_unit_pytorch_256_5fold'+str(k)+'.pt')


    del salt_ID_dataset_train, salt_ID_dataset_val, train_loader, val_loader

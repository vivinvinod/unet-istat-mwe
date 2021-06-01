# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:13:21 2021

@author: vivin
"""

from deep_unet import *
from patch_making import *
#%%
import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#%%
N_CHANNELS = 8
N_CLASSES = 5  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.5, 0.2, 0.1, 0.1, 0.1]
N_EPOCHS = 150
PATCH_SZ = 160   # should be divisible by 160
BATCH_SIZE = 10
TRAIN_SZ = 400  # train size
VAL_SZ = 100    # validation size
#%%
def normalize(img):
    """
    Min-Max Scaler for images
    INPUT: 
        img:
            raw image file
    OUTPUT:
        normalised images
    """
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

def get_model():
    return deep_unet_model(n=N_CLASSES, im_sz = PATCH_SZ, channels=N_CHANNELS, 
                           upconvolution=True, class_weights=CLASS_WEIGHTS)


#%%
weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/deep_unet_model_weights.hdf5'
#%%
#All availiable ids: from "01" to "24"
#change according to number of training images
trainIds = [str(i).zfill(2) for i in range(1, 25)]

#%%

X_DICT_TRAIN = dict()
Y_DICT_TRAIN = dict()
X_DICT_VALIDATION = dict()
Y_DICT_VALIDATION = dict()

print('Reading images')
for img_id in trainIds:
    img_m = normalize(tiff.imread('./data/mband/{}.tif'.format(img_id)).transpose([1, 2, 0]))
    mask = tiff.imread('./data/gt_mband/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
    
    # use 75% of image as train and 25% for validation
    train_xsz = int(3/4 * img_m.shape[0])  
    X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
    Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
    X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
    Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
    print(img_id + ' read')
print('Images were read. start train net')

x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, 
                               n_patches=TRAIN_SZ, sz=PATCH_SZ)
x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, 
                           n_patches=VAL_SZ, sz=PATCH_SZ)
model = get_model()

#check if pretrained model exists
if os.path.isfile(weights_path):
    model.load_weights(weights_path)

model_log = CSVLogger('loss_log_unet_model.csv', append=True, separator=',')

model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', 
                                   save_best_only=True)
#%%

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
          verbose=2, shuffle=True,
          callbacks=[model_checkpoint, model_log],
          validation_data=(x_val, y_val))

#%%
#uncomment this section to plot the loss 
"""
import pandas as pd
import matplotlib.pyplot as plt
#%%
loss_csv = pd.read_csv("loss_log_unet_model.csv", sep=",")
#%%
fig = plt.figure(0)

plt.plot(loss_csv["epoch"],loss_csv["loss"], label="Train Loss")
plt.plot(loss_csv["epoch"],loss_csv["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("ModelLoss.png",dpi = 1000)
"""










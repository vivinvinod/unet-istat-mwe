# -*- coding: utf-8 -*-
"""
Created on Sun May 30 20:55:13 2021

@author: vivin
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from deep_unet import *
#from training import weights_path, normalize, PATCH_SZ, N_CLASSES
weights_path = 'weights'
weights_path += '/deep_unet_model_weights.hdf5'
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

def predict(x, model, patch_sz=160, n_classes=5):
    """
    Function to predict categories of objects on the test image
    
    INPUT:
        x (3D array):
            The image with the channels in the end
        model (keras model):
            pre-trained model from train_unet.py
        patch_sz (Int):
            As defined previously
        n_classes (Int):
            Number of categories that the model was trained on
    
    OUTPUT:
        prediction map
    
    """
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


def picture_from_mask(mask, threshold=0):
    """
    Function to convert the map created into a viewable image.
    """
    colors = {
        0: [150, 150, 150],  # Buildings
        1: [223, 194, 125],  # Roads & Tracks
        2: [27, 120, 55],    # Trees
        3: [166, 219, 160],  # Crops
        4: [116, 173, 209]   # Water
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict
#%%

model = get_model()
model.load_weights(weights_path)
test_id = 'test' #name of hte tif file to predict
img = normalize(tiff.imread('data/mband/{}.tif'.format(test_id)).transpose([1,2,0]))   # make channels last
#%%

for i in range(7):
    if i == 0:  # reverse first dimension
        mymat = predict(img[::-1,:,:], model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
        #print(mymat[0][0][0], mymat[3][12][13])
        print("Case 1",img.shape, mymat.shape)
    elif i == 1:    # reverse second dimension
        temp = predict(img[:,::-1,:], model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
        #print(temp[0][0][0], temp[3][12][13])
        print("Case 2", temp.shape, mymat.shape)
        mymat = np.mean( np.array([ temp[:,::-1,:], mymat ]), axis=0 )
    elif i == 2:    # transpose(interchange) first and second dimensions
        temp = predict(img.transpose([1,0,2]), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
        #print(temp[0][0][0], temp[3][12][13])
        print("Case 3", temp.shape, mymat.shape)
        mymat = np.mean( np.array([ temp.transpose(0,2,1), mymat ]), axis=0 )
    elif i == 3:
        temp = predict(np.rot90(img, 1), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
        #print(temp.transpose([2,0,1])[0][0][0], temp.transpose([2,0,1])[3][12][13])
        print("Case 4", temp.shape, mymat.shape)
        mymat = np.mean( np.array([ np.rot90(temp, -1).transpose([2,0,1]), mymat ]), axis=0 )
    elif i == 4:
        temp = predict(np.rot90(img,2), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
        #print(temp.transpose([2,0,1])[0][0][0], temp.transpose([2,0,1])[3][12][13])
        print("Case 5", temp.shape, mymat.shape)
        mymat = np.mean( np.array([ np.rot90(temp,-2).transpose([2,0,1]), mymat ]), axis=0 )
    elif i == 5:
        temp = predict(np.rot90(img,3), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
        #print(temp.transpose([2,0,1])[0][0][0], temp.transpose([2,0,1])[3][12][13])
        print("Case 6", temp.shape, mymat.shape)
        mymat = np.mean( np.array([ np.rot90(temp, -3).transpose(2,0,1), mymat ]), axis=0 )
    else:
        temp = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
        #print(temp[0][0][0], temp[3][12][13])
        print("Case 7", temp.shape, mymat.shape)
        mymat = np.mean( np.array([ temp, mymat ]), axis=0 )

#%%
map = picture_from_mask(mymat, 0.5)
#%%
mask = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])  # make channels first
map1 = picture_from_mask(mask, 0.5)
#%%
#tiff.imsave('resultmask.tif', (255*mask).astype('uint8'))
tiff.imsave('resultmat.tif', (255*mymat).astype('uint8'))
#%%
tiff.imsave('map.tif', map)
tiff.imsave('map1.tif',map1)

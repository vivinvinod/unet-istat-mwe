# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:34:55 2021

@author: vivin
"""

import random
import numpy as np
"""
This python file contains functions that transform the 
images and return randomized patches.
"""
##############################################################################
def get_rand_patch(img, mask, sz=160):
    """
    Function to return transformed patches of tiff files after random transformations.
    The transformations are 
    1. mirror along first dimension
    2. mirror along second dimension
    3. mirror along diagonal
    4. rotate 90 clockwise
    5. rotate 180 clockwise
    6. rotate 270 clockwise
    
    1/4 times no transformation is applied
    INPUT:
        img (numpy array):
            n-D array with shape (sz.x, sz.y, num_channels)
        mask (binary numpy array):
            n-D binary array with shape (sz.x, sz.y, num_classes)
        sz (Int):
            size of random patch
    
    OUTPUT:
        patch_img:
            image patch of size (sz,sz,num_channels)
        patch_mask:
            mask patch of size (sz,sz,num_channels)
    """
    #check for dimension compatibility
    assert len(img.shape) == 3, "img dimensions not compatible"
    assert img.shape[0] > sz, "img x shape not compatible"
    assert img.shape[1] > sz, "img y shape not compatible"
    assert img.shape[0:2] == mask.shape[0:2], "img mask dimensions incompatible"
    
    #initialise the arrays to store img and maps
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]

    #Apply some random transformations
    #advisable to set seed for reproducible work
    random_transformation = np.random.randint(1,8)
    
    #reverse first dimension
    if random_transformation == 1:
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:]
   
    #Reverse second dimension
    elif random_transformation == 2:
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:]
    
    #Transpose(interchange) first and second dimensions
    elif random_transformation == 3:
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    
    #Rotate 90 clockwise
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    
    #Rotate 180 clockwise
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    
    #Rotate 270 clockwise
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    
    #no transformation 2/8 times (i.e. for 7,8)
    else:
        pass
    
    return patch_img, patch_mask


##############################################################################

def get_patches(x_dict, y_dict, n_patches, sz=160):
    """
    Function to to return image and mask patches
    
    INPUT:
        x_dict (Dictionary):
            Image ID dictionary
        y_dict (Dictionary):
            mask ID dictionary
        n_patches (Int):
            Number of patches to generate
        sz (Int):
            size of the image patch
    
    OUTPUT:
        np.array(x):
            Image patches
        np.array(y):
            Mask patches
    """
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)

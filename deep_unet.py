# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:03:57 2021

@author: vivin
"""
#%%
"""
This python file contains the function that returns a deep U-NET model 
containing up-convolution and up-sampling. The loss function is weighted 
binary-crossentropy. The UNET is built layer by layer and the returned
in the model function.
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras.optimizers import Adam
#from keras.utils import plot_model
from keras import backend as b

#%%
def deep_unet_model(n=5, im_sz=160, channels=8, filter_num=32, 
               growthfactor=2, upconvolution=True,
               class_weights=[0.5, 0.3, 0.1, 0.1, 0.1]):
    """
    Function to return deep-UNET model
    
    INPUT: 
        n (Int):
            Number of categories to identify in the image
        im_size (Int):
            Pixel size of input image
        channels (Int):
            Number of channels in the orthophotos
        filter_num (Int):
            The dimensionality of the output space 
            (i.e. the number of output filters in the convolution).
        growthfactor (Int):
            The scaling of the output dimensions for each layer of the UNET
        upconvolution (Bool):
            Avtivate or deactivate upconvolution
        class_weights (Float Array of length n):
            Should sum to unity else raises assertion error
    
    OUTPUT:
        model (Keras Model):
            TensorFlow backend deep U-NET model
    """
    
    droprate=0.20 #dropout rate
    n_filters = filter_num
    input_layer = Input((im_sz, im_sz, channels))
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    n_filters *= growthfactor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    n_filters *= growthfactor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    n_filters *= growthfactor
    pool3 = BatchNormalization()(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
    pool4_1 = Dropout(droprate)(pool4_1)

    n_filters *= growthfactor
    pool4_1 = BatchNormalization()(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
    pool4_2 = Dropout(droprate)(pool4_2)

    n_filters *= growthfactor
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

    n_filters //= growthfactor
    if upconvolution:
        up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), 
                                             padding='same')(conv5), conv4_1])
    else:
        up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
    up6_1 = BatchNormalization()(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
    conv6_1 = Dropout(droprate)(conv6_1)

    n_filters //= growthfactor
    if upconvolution:
        up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), 
                                             padding='same')(conv6_1), conv4_0])
    else:
        up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
    up6_2 = BatchNormalization()(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
    conv6_2 = Dropout(droprate)(conv6_2)

    n_filters //= growthfactor
    if upconvolution:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), 
                                           padding='same')(conv6_2), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(droprate)(conv7)

    n_filters //= growthfactor
    if upconvolution:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), 
                                           padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(droprate)(conv8)

    n_filters //= growthfactor
    if upconvolution:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), 
                                           padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=input_layer, outputs=conv10)

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = b.mean(b.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return b.sum(class_loglosses * b.constant(class_weights))

    model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy)
    return model

#FIN
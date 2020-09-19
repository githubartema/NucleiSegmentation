import cv2
import os
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.callbacks import EarlyStopping, TensorBoard

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.losses import binary_crossentropy

from model.UNet import UNet
from utils.utils import train_data_loader, dice_coef

import random
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def generator(train_set, val_set, y_train_set, y_val_set, batch_size):
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(train_set, seed=42)
    mask_datagen.fit(y_train_set, seed=42)
    image_generator = image_datagen.flow(train_set, batch_size=batch_size, seed=42)
    mask_generator = mask_datagen.flow(y_train_set, batch_size=batch_size, seed=42)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(val_set, seed=42)
    mask_datagen_val.fit(y_val_set, seed=42)
    image_generator_val = image_datagen_val.flow(val_set, batch_size=batch_size, seed=42)
    mask_generator_val = mask_datagen_val.flow(y_val_set, batch_size=batch_size, seed=42)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator

parser = argparse.ArgumentParser(description='Train UNet')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to the train images', required=True)

args = parser.parse_args()

img_size = 256
batch_size = 32
data_path = args.dir

X_train, Y_train = train_data_loader(data_path, img_size)
train_set, val_set, y_train_set, y_val_set = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
train_generator, val_generator = generator(train_set, val_set, y_train_set, y_val_set, batch_size)

model = UNet(img_size)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

earlystopper = EarlyStopping(patience=5, verbose=1)

model.fit_generator(train_generator, steps_per_epoch=len(train_set)/6, epochs=50, validation_data=val_generator, validation_steps=len(val_set)/batch_size, callbacks=[earlystopper])

model.save("weights/UNet_new_weights.h5")

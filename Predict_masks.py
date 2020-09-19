import cv2
import os
import argparse
import numpy as np

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

from model.UNet import UNet
from utils.utils import test_data_loader, dice_coef
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Evaluate UNet')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it', required=True)

args = parser.parse_args()
img_size = 256
batch_size = 32

if args.dir is not None:
    print("Loading images from directory : ", args.dir)
    X_test, pathes, image_sizes = test_data_loader(args.dir, img_size)

with tf.device('/CPU:0'):
    
    model = UNet(img_size)
    model.load_weights('weights/UNet_087.h5')

    test_prediction = model.predict(X_test, verbose=1)

    for i in range(len(test_prediction)):
        normed_mask = cv2.resize(test_prediction[i], (image_sizes[i][1], image_sizes[i][0]))
        cv2.imwrite(pathes[i], normed_mask*255)#by multiplying by 255 we get to the [0..255] pixel range

print("All the masks are saved to the appropriate directories...")

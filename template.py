from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from RichDataMgr import RichDataMgr

def plot_results(model, test, target):
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(10,5.5))
    plt.ion()
    richOcc = plt.imread('RichOcc.jpg')

    for iEv in range(0, len(test)):
        axs[0].cla()
        axs[0].imshow(model.predict(test[iEv]), origin='lower')
        axs[0].imshow(richOcc, origin='lower', alpha=0.1)
        axs[1].cla()
        axs[1].imshow(target[iEv], origin='lower')
        axs[1].imshow(richOcc, origin='lower', alpha=0.1)
        plt.pause(1)


dim = 145
pad = 20

# get dataset
dataMgr = RichDataMgr('/Volumes/AMS_Disk/DBar/test/test_ML/RichNN/data/training_data.root')
(x_train, y_train), (x_test, y_test) = dataMgr.GetTrainingData(dim, pad, 0.8, 100)
x_train = x_train.astype('float32') / 30.
x_test = x_test.astype('float32') / 30.

print("Data shape before reshaping", x_test[0].shape)

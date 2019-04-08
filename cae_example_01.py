from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
# import argparse
# import os

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
linear_dim = dim + 2*pad

# get dataset
dataMgr = RichDataMgr('/Volumes/AMS_Disk/DBar/test/test_ML/RichNN/data/training_data.root')
(x_train, y_train), (x_test, y_test) = dataMgr.GetTrainingData(145, 20, 0.8, 100)

print("Data shape", x_test.shape)
# x_train = np.reshape(x_train, (-1, linear_dim, linear_dim, 1))
# y_train = np.reshape(y_train, (-1, linear_dim, linear_dim, 1))
# x_test = np.reshape(x_test, (-1, linear_dim, linear_dim, 1))
# y_test = np.reshape(y_test, (-1, linear_dim, linear_dim, 1))

input_img = Input(shape=(linear_dim, linear_dim, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, y_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

plot_results(autoencoder, x_test, y_test)

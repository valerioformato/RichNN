from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np

from RichDataMgr import RichDataMgr

def plot_data(model, x_test, y_true):
    richOcc = plt.imread('RichOcc.jpg')

    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(10,5.5))
    plt.ion()

    y_pred = model.predict(x_test)

    for iEv in range(0, len(x_test)):
        axs[0].cla()
        axs[0].imshow(y_pred[iEv,:,:,0], origin='lower')
        axs[0].imshow(richOcc, origin='lower', alpha=0.1)
        axs[1].cla()
        axs[1].imshow(y_true[iEv,:,:,0], origin='lower')
        axs[1].imshow(richOcc, origin='lower', alpha=0.1)
        plt.pause(5)


def my_loss(y_true, y_pred): #controlla l'ordine
    # Shapes
    # y_true --> NBatch * NPixX * NPixY * 2 --> [0 il target, 1 la mask]
    # y_pred --> NBatch * NPixX * NPixY
    target = y_true[:,:,:,0]
    mask = y_true[:,:,:,1]
    pred = y_pred[:,:,:,0]
    loss_vals = K.binary_crossentropy(target, pred)*mask
    return K.sum(K.sum(loss_vals, axis=-1), axis=1)


dataMgr = RichDataMgr('/Volumes/AMS_Disk/DBar/test/test_ML/RichNN/data/training_data.root')
(X_train, y_train), (X_test, y_test) = dataMgr.GetTrainingData(145, 20, 0.8, 200)
print X_train.shape[1], X_train.shape[2]

input_shape = X_train[0].shape

#plot the first image in the dataset
# plt.imshow(X_train[0])
# plt.show()

#create model
model = Sequential()
#add model layers
model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(1 , kernel_size=5, padding='same', activation='relu'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss=my_loss)

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

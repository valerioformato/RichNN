from time import time

import tensorflow as tf
import keras.callbacks
from keras.datasets import mnist
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


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.75
session = tf.Session(config=config)

processing_batch_size = 32
dim, pad = 145, 20

trainDataMgr = RichDataMgr('training_data.root', processing_batch_size, True)
testDataMgr  = RichDataMgr('training_data.root', processing_batch_size, False)

trainDataMgr.SetShape(dim, pad)
testDataMgr.SetShape(dim, pad)

(x0_train, y0_train) = trainDataMgr[0]
input_shape = x0_train[0].shape
print("Input shape is ", input_shape)

#plot the first image in the dataset
# plt.imshow(X_train[0])
# plt.show()

#create model
model = Sequential()
#add model layers
model.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(8, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(1, kernel_size=5, padding='same', activation='relu'))

modelCallbacks = []
modelCallbacks.append( keras.callbacks.TensorBoard(log_dir='./Graph/{}'.format(time()), histogram_freq=0, write_graph=True, write_images=True, update_freq=100*processing_batch_size) )
modelCallbacks.append( keras.callbacks.TerminateOnNaN() )
modelCallbacks.append( keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1) )

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss=my_loss, metrics=['accuracy'])

train_Steps = len(trainDataMgr)
test_Steps = len(testDataMgr)

#train the model
model.fit_generator(
    generator=trainDataMgr,
    steps_per_epoch=train_Steps,
    validation_data=testDataMgr,
    validation_steps=test_Steps,
    callbacks=modelCallbacks,
    epochs=5
    )

model.save('RichNN.h5')
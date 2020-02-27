import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Layer
from keras.regularizers import l2
from keras import backend 
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def exercise3():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = (X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32'))/255
    X_test = (X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32'))/255
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    image_generator = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.05, 
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        vertical_flip=False)

    randidx = np.random.randint(len(X_train), size=10000)
    x_augmented = X_train[randidx].copy()
    y_augmented = Y_train[randidx].copy()
    x_augmented = image_generator.flow(x_augmented, np.zeros(10000), batch_size=10000, shuffle=False).next()[0]
    X_train = np.concatenate((X_train, x_augmented))
    Y_train = np.concatenate((Y_train, y_augmented))

    X_train_train = X_train[0:62000]
    X_validate = X_train[62000:70000]
    Y_train_train = Y_train[0:62000]
    Y_validate = Y_train[62000:70000]

    model = Sequential()
    model.add(Conv2D(25, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(50, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='linear', W_regularizer=l2(0.01)))

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto'),
              ModelCheckpoint(filepath='exercise3.h5', monitor='val_loss', save_best_only=True)]

    model.compile(loss=keras.losses.hinge, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(X_train_train, Y_train_train, epochs=20, verbose=1, callbacks=callbacks, validation_data=(X_validate, Y_validate))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

exercise3()
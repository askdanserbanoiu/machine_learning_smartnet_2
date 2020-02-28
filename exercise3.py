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
import glob
import cv2


def import_test_custom(n):
    image_list = []
    labels = []
    for i in n:
        f = 0
        for filename in glob.glob('exercise3/'+str(i)+'/*.png'): 
            file = cv2.imread(filename)
            file = cv2.resize(file, (28, 28))
            file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
            file = file.reshape((28, 28,1))
            image_list.append(file)
            f = f + 1
        labels.extend([i]*f)
    return image_list, labels

def exercise3():

    images, labels = import_test_custom(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    labels = np_utils.to_categorical(labels, 10)


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
    score = model.evaluate(np.array(images), np.array(labels), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

exercise3()
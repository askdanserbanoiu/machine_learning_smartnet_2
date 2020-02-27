import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as sio
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def get_layer_outputs(img, model):
    test_image = img
    outputs    = [layer.output for layer in model.layers]          # all layer outputs
    comp_graph = [K.function([model.input]+ [K.learning_phase()], [output]) for output in outputs]  # evaluation functions

    # Testing
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs


def plot_layer_outputs(img, model, layer_number):    
    layer_outputs = get_layer_outputs(img, model)

    x_max = layer_outputs[layer_number].shape[0]
    y_max = layer_outputs[layer_number].shape[1]
    n     = layer_outputs[layer_number].shape[2]

    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))

    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[layer_number][x][y][i]


    for img in L:
        plt.figure()
        plt.imshow(img, interpolation='nearest')



def exercise2_b():   


    train_data = sio.loadmat(os.path.join(os.path.join(os.getcwd(), "svhn"), 'train.mat'))
    test_data = sio.loadmat(os.path.join(os.path.join(os.getcwd(), "svhn"), 'test.mat'))

    X_train = train_data['X']
    Y_train = train_data['y']
    Y_train[Y_train == 10] = 0

    X_test = test_data['X']
    Y_test = test_data['y']
    Y_test[Y_test == 10] = 0


    X_train = (X_train.transpose(3, 0, 1, 2)/255).astype(np.float32)
    X_test = (X_test.transpose(3, 0, 1, 2)/255).astype(np.float32)

    Y_train = keras.utils.to_categorical(Y_train.flatten(), 10)
    Y_test = Y_test.flatten()


    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_train_train = X_train[7326:len(X_train)]
    X_validation = X_train[0:7326]
    Y_train_train = Y_train[7326:len(Y_train)]
    Y_validation = Y_train[0:7326]

    model = models.Sequential()

    model.add(Convolution2D(9, (3, 3), padding='same', activation='relu', input_shape=(32,32,3),data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(36, (3, 3), padding='same', activation='relu',  data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Convolution2D(49, (3, 3), padding='same', activation='relu',data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto'),
                ModelCheckpoint(filepath='exercise2_a.h5', monitor='val_loss', save_best_only=True)]

    model.fit(X_train_train, Y_train_train, validation_data=(X_validation, Y_validation), callbacks=callbacks, batch_size=500, epochs=20)

    Y_pred = model.predict(X_test, verbose=0).argmax(axis=-1)

exercise2_b()


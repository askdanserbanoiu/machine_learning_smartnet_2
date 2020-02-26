import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import keras.layers as l
import keras.optimizers as o
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import  Sequential

#biased because test data is used as validation set. When testing occurs accuracy appears higher than it actually is

def exercise1_b(activation_functions, layers):
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = ((X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')/255)
    X_test = ((X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32'))/255)
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    results = []

    for af in activation_functions:

        for layer in layers:
        
            model = Sequential()
            model.add(Flatten())

            for n in range(0, layer):
                model.add(Dense(32, activation=af))

            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer=o.SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_train, Y_train, epochs=3, validation_data = (X_test, Y_test))

            score, acc = model.evaluate(X_test, Y_test, verbose=0)

            results.append([af, layer, str(acc*100) + "%"])

    print(results)


exercise1_b(['relu', 'tanh', 'sigmoid'], [5, 20, 40])
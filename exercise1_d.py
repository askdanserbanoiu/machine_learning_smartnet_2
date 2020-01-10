import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import keras.layers as l
import keras.optimizers as o
import keras.models as m
import keras

from keras.layers import Dense, Flatten, Activation
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import  Sequential
from matplotlib import pyplot as plt
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

def lecun_activation(x):
    return (K.tanh((2/3) * x) * 1.7159) + 0.01*x

get_custom_objects().update({'lecun': Activation(lecun_activation)})

results = []

for af in ['lecun', 'tanh']:
    
    for l in [5, 20, 40]:
    
        model = Sequential()
        model.add(Flatten())

        for k in range(0, l):
        
            model.add(Dense(32, activation='lecun'))


        model.add(Dense(10, activation='softmax'))

        sgd = o.SGD(lr=0.01)

        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train,Y_train, epochs=3, validation_data = (X_test, Y_test))

        score = model.evaluate(X_test, Y_test, verbose=0)

        results.append([af,l, score[1]])

print(results)

import numpy as np
import matplotlib.pyplot as plt
import keras.layers as l
import keras.optimizers as o
import keras.models as m
import keras

from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import  Sequential

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)


from matplotlib import pyplot as plt
plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print(Y_train[0])

model = Sequential()






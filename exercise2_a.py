# Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os


SVHN_directory = os.path.join(os.path.join(os.getcwd(), os.path.join("svhn", "train.mat")))
# load .mat file
data_raw = loadmat(SVHN_directory)
data = np.array(data_raw['X'])
# make correct shape
data = np.moveaxis(data, -1, 0)
#print(data.shape)
#plt.imshow(data[2])

#plt.show()
#print(data[0])

plt.show()

labels = data_raw['y']

#print(int(labels[2]))

x_train=data
y_train=labels

# Preprocess input data

X_train = x_train.astype('float32')

X_train /= 255

#print(X_train[2])

# Preprocess class labels
Y_train = np_utils.to_categorical(y_train.reshape([-1, 1])) # need of understanding how reshape([-1, 1]) works
print(Y_train[2])
 
# Define model architecture
model = Sequential()


"""
Because of error:
-------------------------------------------------------------------------------- 
ValueError: Negative dimension size caused by subtracting 3 from 1 for 
'conv2d_2/convolution' (op: 'Conv2D') with input shapes: [?,1,28,28],[3,3,28,32].
--------------------------------------------------------------------------------
 input_shape=(1,28,28),data_format='channels_first' is added
"""

model.add(Convolution2D(9, (3, 3), padding='same', activation='relu', input_shape=(32,32,3),data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(36, (3, 3), padding='same', activation='relu',  data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(49, (3, 3), padding='same', activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='relu'))
 
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=2, verbose=1)
 
# Evaluate model on test data
#score = model.evaluate(X_test, Y_test, verbose=0)

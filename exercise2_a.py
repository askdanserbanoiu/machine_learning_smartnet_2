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
print(data.shape)
plt.imshow(data[0])
<<<<<<< HEAD
#plt.show()
#print(data[0])
=======
plt.show()
#print(data[31][31])
>>>>>>> 28c04e10df1ac523ec083de8e24ae437032436e7
labels = data_raw['y']
print(labels.shape)
print(labels[0])


# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
#print(X_train[0])

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#print(X_train[0])
""" 
# Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 
# Define model architecture
model = Sequential()
"""

"""
Because of error:
-------------------------------------------------------------------------------- 
ValueError: Negative dimension size caused by subtracting 3 from 1 for 
'conv2d_2/convolution' (op: 'Conv2D') with input shapes: [?,1,28,28],[3,3,28,32].
--------------------------------------------------------------------------------
 input_shape=(1,28,28),data_format='channels_first' is added
"""
"""
model.add(Convolution2D(9, (3, 3), padding='same', activation='relu', input_shape=(1,28,28),data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(36, (3, 3), padding='same', activation='relu',  data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(49, (3, 3), padding='same', activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
 
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=2, verbose=1)
 
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
"""
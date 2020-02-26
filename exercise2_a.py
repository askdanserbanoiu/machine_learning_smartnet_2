import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

def encoder(y):  
    y = (np.arange(10) == y[:, np.newaxis]).astype(np.float32)
    return y

train_data = sio.loadmat(os.path.join(os.path.join(os.getcwd(), "svhn"), 'train.mat'))
test_data = sio.loadmat(os.path.join(os.path.join(os.getcwd(), "svhn"), 'test.mat'))

X_train = train_data['X']
Y_train = train_data['y']
X_test = test_data['X']
Y_test = test_data['y']

X_train = (X_train.transpose(3, 0, 1, 2)/255).astype(np.float32)
X_test = (X_test.transpose(3, 0, 1, 2)/255).astype(np.float32)
Y_train = encoder(Y_train.flatten())
Y_test = encoder(Y_test.flatten())

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

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=callbacks, batch_size=500, epochs=10)

score, acc = model.evaluate(X_test, Y_test, verbose=0)

#plt.figure(figsize=(12, 8))
#cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=flat_array)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
#sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True);
print(acc)

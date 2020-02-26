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

def one_hot_encoder(y):  
    y = (np.arange(10) == y[:, np.newaxis]).astype(np.float32)
    return y

def change_dim_x(x_data):
    new_x = (x_data.transpose(3, 0, 1, 2)/255).astype(np.float32)
    return new_x


train_data = sio.loadmat(os.path.join(os.path.join(os.getcwd(), "svhn"), 'train.mat'))
test_data = sio.loadmat(os.path.join(os.path.join(os.getcwd(), "svhn"), 'test.mat'))

x_train = train_data['X']
y_train = train_data['y']

x_test = test_data['X']
y_test = test_data['y']

x_train = change_dim_x(x_train)
x_test = change_dim_x(x_test)

y_train = one_hot_encoder(y_train.flatten())
y_test = one_hot_encoder(y_test.flatten())

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

model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=callbacks, batch_size=500, epochs=10)

score, acc = model.evaluate(x_test, y_test, verbose=0)

#plt.figure(figsize=(12, 8))
#cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=flat_array)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
#sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True);
print(acc)

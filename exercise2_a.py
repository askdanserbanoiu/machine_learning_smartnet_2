import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import scipy.io as sio
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint


def print_figure(figure_name):
    
    figure_path = os.path.join(os.path.join(os.getcwd(), "figures"))
    
    if os.path.isdir(figure_path):
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    else:
        os.mkdir(figure_path)
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    
    return
  
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def compute_confusion_matrix(true, pred):
  K = len(np.unique(true)) 
  cf_matrix = np.zeros((K, K))
  for i in range(len(true)):
    cf_matrix[true[i]][pred[i]] += 1
  for i in range(len(cf_matrix)):
        for j in range(len(cf_matrix[i])):
                cf_matrix[i][j] = (cf_matrix[i][j]/np.amax(cf_matrix))*100
  return cf_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt_cm = np.array(cm)    
    plt.imshow(plt_cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = plt_cm.max() / 2.
    for i, j in itertools.product(range(plt_cm.shape[0]), range(plt_cm.shape[1])):
        plt.text(j, i, "{0:.1f}".format(plt_cm[i, j]), horizontalalignment="center", color="white" if plt_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')
    plt.legend(fontsize='small')
    plt.gcf().subplots_adjust(bottom=0.15)
    print_figure("exercise2_a") 


def exercise2_a():   
  train_data = sio.loadmat(os.path.join(os.path.join(os.getcwd(), "svhn"), 'train_32x32.mat'))
  test_data = sio.loadmat(os.path.join(os.path.join(os.getcwd(), "svhn"), 'test_32x32.mat'))

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

  model.fit(X_train_train, Y_train_train, validation_data=(X_validation, Y_validation), callbacks=callbacks, batch_size=500, epochs=3)

  Y_pred = model.predict(X_test, verbose=0).argmax(axis=-1)

  confusion = compute_confusion_matrix(Y_test, Y_pred)

  plot_confusion_matrix(confusion, ["1","2","3","4","5","6","7","8","9"])

exercise2_a()
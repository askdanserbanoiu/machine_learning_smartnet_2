import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import keras.layers as l
import keras.optimizers as o
from keras.layers import Dense, Flatten, Activation
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import  Sequential
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K

def lecun_activation(x):
    return (K.tanh((2/3) * x) * 1.7159) + 0.01*x

def print_figure(figure_name):
    
    figure_path = os.path.join(os.path.join(os.getcwd(), "figures"))
    
    if os.path.isdir(figure_path):
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    else:
        os.mkdir(figure_path)
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    
    return

def get_gradients(model, inputs, outputs):
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad

def max_gradients(gradients):
    max_grad_per_layer = []                    
    for i in range(0, len(gradients), 2):
        max_grad_per_layer.append(np.max([np.max(gradients[i]), np.max(gradients[i+1])]))
    return max_grad_per_layer

def column(matrix, i):
    return [row[i] for row in matrix]

def exercise1_d(activation_functions, layers):
    get_custom_objects().update({'lecun': Activation(lecun_activation)})

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = ((X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')/255)
    X_test = ((X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32'))/255)
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    X_train_train = X_train[0:55000]
    X_validate = X_train[55000:60000]
    Y_train_train = Y_train[0:55000]
    Y_validate = Y_train[55000:60000]
    
    results = []

    for af in activation_functions:

        for layer in layers:
        
            model = Sequential()
            model.add(Flatten())

            for n in range(0, layer):
                model.add(Dense(32, activation=af))

            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer=o.SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_train,Y_train, epochs=3, validation_data = (X_test, Y_test))

            score, acc = model.evaluate(X_test, Y_test, verbose=0)

            results.append([af, layer, str(acc*100) + "%"])

    print(results)

    for layer in layers:

        for af in activation_functions:

            #training part

            model = Sequential()
            model.add(Flatten())

            for n in range(0, layer):
                model.add(Dense(32, activation=af))

            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer=o.SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_train_train,Y_train_train, epochs=3, validation_data = (X_validate, Y_validate))

            score, acc = model.evaluate(X_test, Y_test, verbose=0)

            results.append([af, layer, acc])

            #gradients part
        			
            model = Sequential()
            model.add(Flatten())

            for i in range(0, layer):
                model.add(Dense(32, activation=af))

            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer=o.SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
			
            model.fit(X_train,Y_train, epochs=3, batch_size= 64)
            
            max_grad_per_layer = max_gradients(get_gradients(model, X_train, Y_train))

            score, acc = model.evaluate(X_test, Y_test, verbose=0)
                        
            l=str(af)+' , accuracy='+str("%.3f" %acc)

            plt.plot(range(1, layer + 2), max_grad_per_layer, 'o', label=l)
                        
            
        plt.title("Max gradients per layer for " + str(layer + 1) + " layers")
        plt.legend(fontsize='small')
        print_figure("exercise1_d_gradients_" + str(layer + 1)) 
        plt.figure()  


exercise1_d(['lecun', 'tanh'], [5, 20, 40])

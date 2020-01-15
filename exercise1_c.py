import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import keras.layers as l
import keras.optimizers as o
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import  Sequential
from keras import backend as K

def print_figure(figure_name):
    
    figure_path = os.path.join(os.path.join(os.getcwd(), "figures"))
    
    if os.path.isdir(figure_path):
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    else:
        os.mkdir(figure_path)
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    
    return

def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad

def max_grads(grads):
    result = []
    for i in grads:
        result.append(np.max(i))
    return result


def get_gradients(model, inputs, outputs):
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + weight)
    print(len(output_grad))
    grads=np.array(output_grad)

    result = []
    for layer in range(len(model.layers)):
        if model.layers[layer].__class__.__name__ == 'Dense':
            print(grads[layer].shape)
            result.append(np.max(grads[layer]))

    print(result)

    return result


def exercise1_c(activation_functions, layers):
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)


    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    axis = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    index = 0

    for af in activation_functions:

        for layer in layers:
        
            model = Sequential()
            model.add(Flatten())

            for n in range(0, layer):
                model.add(Dense(32, activation=af))

            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer=o.SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_train,Y_train, epochs=3, batch_size= 200, validation_data = (X_test, Y_test))

            X = range(layer + 1)
            Y = max_grads(get_weight_grad(model, X_train, Y_train))

            axis[index].plot(X, Y, 'o')
            axis[index].set_title((str(af) + str(layer)))

            index = index + 1

    print_figure("exercise1_c_gradients")
    plt.show()

exercise1_c(['relu', 'tanh', 'sigmoid'], [5, 20, 40])
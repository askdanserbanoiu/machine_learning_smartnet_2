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

def get_gradients(model, inputs, outputs):
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + weight)
    
    return np.array(output_grad)


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

    fig,(ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle('layer depth vs. max gradient')
    axis=[ax1, ax2, ax3]
    
    index=0
    
    for af in activation_functions:
      
        for layer in layers:
            
            print("Model : " + str(af) + " - " + str(layer) + " layers")
			
            model = Sequential()
            model.add(Flatten())

            for n in range(0, layer):
                model.add(Dense(32, activation=af))

            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer=o.SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
			
            model.fit(X_train,Y_train, epochs=3, batch_size= 64)
            
            grads = get_gradients(model, X_train, Y_train)

            max_gradient_layer=[]
            
            for i,_ in enumerate(grads):
                if(i%2==0):
                    max_gradient_layer.append(np.max(grads[i]))

            score = model.evaluate(X_test, Y_test, verbose=0)
            
            depth=range(1,layer+1)
            
            l=str(af)+' , acc.='+str("%.3f" %score[1])
            
            axis[index].plot(depth, max_gradient_layer[0:len( max_gradient_layer)-1],'o',label=l)
            
            axis[index].set_title(str(layer)+" layers")
            
            axis[index].legend(fontsize='small')
            
            index+=1
            
        index=0


    print_figure("exercise1_c_gradients")
    plt.show()

exercise1_c(['relu', 'tanh', 'sigmoid'], [5, 20, 40])
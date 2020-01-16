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


def get_gradients(model, inputs, outputs):
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + weight)
    #print(len(output_grad))
    grads_a=np.array(output_grad)
    
    return grads_a

def get_layer_output_grad(model, inputs, outputs,layer):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
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
	
def exercise1_c(af, layers):
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #plt.imshow(X_train[0])

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    	
    model = Sequential()
    model.add(Flatten())
    for n in range(0, layers):
        model.add(Dense(32, activation=af))
		
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=o.SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train,Y_train, epochs=3, batch_size=64)
    #mini_batch = model.train_on_batch(X_train[0:1], Y_train[0:1])
    model.summary()
	#max_weight_grads = get_max_gradient_per_layer(get_weight_grad(model, X_test, Y_test))
    grads = get_gradients(model, X_train, Y_train)
    print("shape = "+str(grads.shape))
    max_gradient_layer=[]
    for i,_ in enumerate(grads):
        if(i%2==0):
            print(grads[i].shape)
            max_gradient_layer.append(np.max(grads[i]))
    print(max_gradient_layer)
    score = model.evaluate(X_test, Y_test, verbose=0)
    depth=range(1,layers+1)
    #print(score)
    _=plt.figure()
    _=plt.plot(0,label='accuracy = '+str("%.3f" %score[1]), color='white')
    _=plt.plot(depth, max_gradient_layer[0:len( max_gradient_layer)-1],'o' )
    _= plt.legend(fontsize='small')
    _=plt.show()	
   

	#results.append([af, layer, score[1], len(max_weight_grads)])

    #print(results)

#change activation function or number of layers below 
exercise1_c('relu', 5)
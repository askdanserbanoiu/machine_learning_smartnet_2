import numpy as np
import matplotlib.pyplot as plt
from math import exp

def sigmoid(x):
    return(1/(1+exp(-x)))
	
def d_sigmoid(x):
    return(exp(-x)*(sigmoid(x)**2))
	
def tanh(x):
    return (exp(x)-exp(-x))/(exp(x)+exp(-x))

def d_tanh(x):
    return (1-(tanh(x)**2))
	
def relu(x):
    return (np.maximum(0,x))
	
def d_relu(x):
    if(x>0): return 1
    else: return 0
	
def values_of(function,x):
    f = np.vectorize(function) 
    y = f(x)
    return y
	
x = np.arange(-5, 5, 0.01)

fig,axs = plt.subplots(2,2,gridspec_kw={'hspace': 0.3})
fig.suptitle('activation functions and their derivatives')

axs[0,0].plot(x, values_of(sigmoid,x), label="sigmoid(x)")
axs[0,0].plot(x, values_of(d_sigmoid,x), label="sigmoid\'(x)")
axs[0,0].set_title("sigmoid")
axs[0,0].legend(fontsize='x-small')

axs[0,1].plot(x, values_of(tanh,x), label='tanh(x)')
axs[0,1].plot(x, values_of(d_tanh,x), label="tanh\'(x)")
axs[0,1].set_title("tanh")
axs[0,1].legend(fontsize='x-small')

axs[1,0].plot(x, values_of(relu,x))
axs[1,0].set_title("relu")
axs[1,0].axis([-5,5,-0.1,2])

axs[1,1].plot(x, values_of(d_relu,x), ls='', color='darkorange', marker='.',  markeredgewidth=0.00001, markersize=1.9)
axs[1,1].axis([-5,5,-0.1,2])
axs[1,1].plot(0,0, 'o', color='darkorange', markerfacecolor='white')
axs[1,1].plot(0,1, 'o', color='darkorange', markerfacecolor='white')
axs[1,1].set_title("Derivative of relu")	

_=plt.show()


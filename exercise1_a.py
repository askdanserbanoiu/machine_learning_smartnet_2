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
    return (max(0,x))
	
def d_relu(x):
    if(x>0): return 1
    else: return 0
	
x_space = np.linspace(-5, 5, 2000)
dx_relu=x_space.copy()
	
sigmoid_y=[]
Dsigmoid_y=[]
tanh_y=[]
Dtanh_y=[]
relu_y=[]
Drelu_y=[]

for x in x_space:
    sigmoid_y.append(sigmoid(x))
    Dsigmoid_y.append(d_sigmoid(x))
    tanh_y.append(tanh(x))
    Dtanh_y.append(d_tanh(x))
    relu_y.append(relu(x))
    if(x==0):
	    dx_relu.remove(x) #Derivative of relu is not defined at zero 
    Drelu_y.append(d_relu(x))	

fig,axs = plt.subplots(2,2,gridspec_kw={'hspace': 0.3})
fig.suptitle('activation functions and their derivatives')

axs[0,0].plot(x_space, sigmoid_y, label='sigmoid(x)')
axs[0,0].plot(x_space, Dsigmoid_y, label="sigmoid\'(x)")
axs[0,0].set_title("sigmoid")
axs[0,0].legend(fontsize='x-small')

axs[0,1].plot(x_space, tanh_y, label='tanh(x)')
axs[0,1].plot(x_space, Dtanh_y, label="tanh\'(x)")
axs[0,1].set_title("tanh")
axs[0,1].legend(fontsize='x-small')

axs[1,0].plot(x_space, relu_y)
axs[1,0].set_title("relu")
axs[1,0].axis([-5,5,-0.1,2])

axs[1,1].plot(dx_relu, Drelu_y, ls='', color='darkorange', marker='.',  markeredgewidth=0.00001, markersize=1.9)
axs[1,1].axis([-5,5,-0.1,2])
axs[1,1].plot(0,0, 'o', color='darkorange', markerfacecolor='white')
axs[1,1].plot(0,1, 'o', color='darkorange', markerfacecolor='white')
axs[1,1].set_title("Derivative of relu")
	
_=plt.show()


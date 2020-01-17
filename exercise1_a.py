import numpy as np
import matplotlib.pyplot as plt
from math import exp

x_space = np.linspace(-5, 5, 2000)

fig,axs = plt.subplots(2,2,gridspec_kw={'hspace': 0.3})
fig.suptitle('activation functions and their derivatives')
#fig.legend(fontsize='large')
	
sigmoid=[]
d_sigmoid=[]
for x in x_space:
    sigmoid.append(1/(1+exp(-x)))
    d_sigmoid.append(exp(-x)*(sigmoid[-1]**2))
axs[0,0].plot(x_space, sigmoid, label='sigmoid(x)')
axs[0,0].plot(x_space, d_sigmoid, label="sigmoid\'(x)")
axs[0,0].set_title("sigmoid")
axs[0,0].legend(fontsize='x-small')

tanh=[]
d_tanh=[]
for x in x_space:
    tanh.append((exp(x)-exp(-x))/(exp(x)+exp(-x)))
    d_tanh.append(1-(tanh[-1]**2))
axs[0,1].plot(x_space, tanh, label='tanh(x)')
axs[0,1].plot(x_space, d_tanh, label="tanh\'(x)")
axs[0,1].set_title("tanh")
axs[0,1].legend(fontsize='x-small')

relu=[]
d_relu=[]
dx_space=x_space.copy()
for x in dx_space:
    relu.append(max(0,x))
    if(x>0):
        d_relu.append(1)
    elif(x<0):
        d_relu.append(0)
    else:
        dx_space.remove(x)
axs[1,0].plot(x_space, relu)
axs[1,0].set_title("relu")
axs[1,0].axis([-5,5,-0.1,2])
axs[1,1].plot(dx_space, d_relu, ls='', color='darkorange', marker='.',  markeredgewidth=0.00001, markersize=1.9)
axs[1,1].axis([-5,5,-0.1,2])
axs[1,1].plot(0,0, 'o', color='darkorange', markerfacecolor='white')
axs[1,1].plot(0,1, 'o', color='darkorange', markerfacecolor='white')
axs[1,1].set_title("Derivative of relu")
	
_=plt.show()


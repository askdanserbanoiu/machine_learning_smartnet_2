import numpy as np
import matplotlib.pyplot as plt
from math import exp
import os


def print_figure(figure_name):
    
    figure_path = os.path.join(os.path.join(os.getcwd(), "figures"))
    
    if os.path.isdir(figure_path):
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    else:
        os.mkdir(figure_path)
        plt.savefig(os.path.join(figure_path, figure_name), quality=99)
    
    return


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
	
def LeCun(x):
    return (1.7159*tanh((2/3)*x)+0.01*x)
	
def d_LeCun(x):
   return (1.7159*d_tanh(x)*(2/3)+0.01)
   
def values_of(function, x):
    f = np.vectorize(function) 
    y = f(x)
    return y
	
x = np.arange(-5, 5, 0.01)

plt.plot(x, values_of(sigmoid,x), label="sigmoid(x)")
plt.plot(x, values_of(d_sigmoid,x), label="sigmoid\'(x)")
plt.title("Sigmoid activation function and its derivative")
plt.legend(fontsize='x-small')
print_figure("exercise1_a_sigmoid") 
plt.figure()  

plt.plot(x, values_of(tanh,x), label='tanh(x)')
plt.plot(x, values_of(d_tanh,x), label="tanh\'(x)")
plt.title("Tanh activation function and its derivative")
plt.legend(fontsize='x-small')
print_figure("exercise1_a_tanh") 
plt.figure() 

plt.plot(x, values_of(relu,x), label='relu(x)')
plt.plot(x, values_of(d_relu,x), label='relu\'(x)', color='darkorange', marker='.',  markeredgewidth=0.00001, markersize=1.9)
plt.plot(0,0, 'o', color='darkorange', markerfacecolor='white')
plt.plot(0,1, 'o', color='darkorange', markerfacecolor='white')
plt.title("Relu activation function and its derivative")
plt.legend(fontsize='x-small')
print_figure("exercise1_a_relu") 
plt.figure() 

plt.figure(5)
plt.plot(x, values_of(LeCun,x), label="LeCun(x)" )
plt.plot(x, values_of(d_LeCun,x), label="LeCun\'(x)" )
plt.title("LeCun activation function and its derivative")
plt.legend(fontsize='x-small')
print_figure("exercise1_d_lecun") 
plt.figure() 



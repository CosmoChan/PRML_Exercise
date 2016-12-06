from base_function import Sigmoid,Linear
import numpy as np


def W_init(hiddensize,X):
	'''initial the weight of the network

	-------
	Parameters
	hiddensize : list = shape(1,size)
				size is the number of hidden layers - 1

	X : matrix = shape(784+1,60000)
				784+1 : features with bias per sample
				60000 : samples
	-------
	Returns
	W : list = shape(1,W_i * size)
				W_i : matrix = shape(n,1)
	'''
	
	w = []
	outputsize = 10
	inputsize,nsamples = np.shape(X)
	for i in range(len(hiddensize)):
		if i == 0:
			temp = np.random.random(size=(inputsize,hiddensize[0]))
			w.append(temp)
		else:
			temp = np.random.random(size=(hiddensize[i-1],hiddensize[i]))
			w.append(temp)
	temp = np.random.random(size=(hiddensize[-1],outputsize))
	w.append(temp)
	return w

def Activation_function(funclist):
	'''initial the Activation function of layers

	-------
	Parameters
	funclist : list = shape(1,size)
				size is the number of hidden layers
				list contains the name of function
				
				Sigmoid: 1/(1+e^(-x))
				Linear: x
	-------
	Returns
	AF : Activation function
	'''
	AF = funclist
	return AF
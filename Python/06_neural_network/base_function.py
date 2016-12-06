import numpy as np
def Sigmoid(X):
	'''Motivate function,Y = 1/(1+e^(-x))

	Parameters
	---------
	X : float
		the linear combination to be motivated
	---------

	Returns
	---------
	Y : float
		data motivated
	---------
	'''
	Y = 1.00/(1.00+np.exp(-X))
	return Y

def Linear(X):
	'''Motivate function,Y = X

	Parameters
	---------
	X : float
		the linear combination to be motivated
	---------

	Returns
	---------
	Y : float
		data motivated
	---------
	'''
	Y = X
	return Y

def show_W(W):
	'''Visit the shape of layer

	Parameters
	---------
	W : Weight of nnet
			list = shape(1,W_i * size)
			W_i : matrix = shape(n,1)
	---------
	'''
	for i in range(len(W)):
		print 'layer_',i,':',np.shape(W[i])

def add_dummy_variable(X):
	''' Add dummy variable to X

	Parameters
	----------
	X : list ,shape(60000,784)
		60000 : the number of samples
		784 : the features of sample
	----------

	Returns
	----------
	X : list , shape(60000,785)
		60000 : the number of samples
		784 : the features of sample and bias
	----------
	'''
	n, m = np.shape(X)
	dummy_variable = np.ones((n,1))
	X = np.hstack((dummy_variable, X))
	return X

def data_T(X,y):
	return np.array(X).T.tolist(),np.array(y).T.tolist()

def Softmax(Y_hat):
	'''map X from [-\infty,+\infty] to [0,1],and set the max value 1,the other value 0

	Parameters
	---------
	Y_hat : matrix ,shape = (60000,10)
		60000 : the number of samples
		10 : the result of the output layer 
	---------
	
	Returns
	---------
	Y_H : matrix ,shape = (60000,10)
		60000 : the number of samples
		1 : result , cast in range(0,10)
	---------

	'''
	n,m = np.shape(Y_hat)  #n=60000,m=10
	exp_Y = np.exp(Y_hat).T
	sum_exp_Y = np.sum(exp_Y,axis=0)
	Y_hat = exp_Y/sum_exp_Y
	Y_hat = Y_hat.T
	return Y_hat


def OneofK(y):
	'''code y in one of k

	Parameters
	---------
	y : list shape(60000,1)
	---------
	Returns
	---------
	Y : matrix shape(60000,10)
		60000 : the number of samples
		10 : 1ofK code.
	---------
	'''
	n = len(y)
	Y = np.zeros([n,10])
	for i in range(n):
		Y[i,y[i]] = 1.00
	return Y


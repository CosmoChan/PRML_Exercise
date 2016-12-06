import numpy as np
from base_function import Sigmoid,Linear,show_W,add_dummy_variable,data_T,Softmax,OneofK


def nnet_calculation(W,AF,X,y):
	'''calculation the result 

	Parameters
	-------
	W : Weight of nnet
			list = shape(1,W_i * size)
			W_i : matrix = shape(n,m)
			n : previous layer size
			m : next layer size
	AF : Activation function
			list = shape(1,size)
			list contains the name of function

	X : list = shape(785,60000)
			785 : number of features
			60000 : number of samples
	
	y : list = shape(60000,10)
			10 : one of K
			60000 : number of samples
	-------

	Returns
	-------
	Y_hat : matrix = shape(10,60000)
			prediction of X 
	-------
	'''
	A = []		#A will be shape(n)
	n,m = np.shape(X)
	X = np.array(X).T   # X = shape(60000,785)
	y = np.array(y)   # y = shape(60000,10)


	#====calculating A
	layersize = len(W)		
	for i in range(layersize):
		if i == 0:
			A_i = X   
			A.append(A_i)   # A1
			Y_hat = A_i
		else:
			A_i = Sigmoid(np.dot(Y_hat, W[i-1]))
			A.append(A_i)
			Y_hat = A_i
 	TEST = np.dot(Y_hat,W[layersize-1])
	A_n = Softmax(np.dot(Y_hat,W[layersize-1]))
	A.append(A_n)      #  An 
	Y_hat = A_n  
	E1 = np.abs(Y_hat-y)
	ERROR = 1.0/(2*m)*sum(sum(E1))
	return ERROR,Y_hat,A

def Back_propagation(W,eta,Y_hat,y,A):
	'''Back_propagation

	Parameters
	-------
	W : Weight of nnet
			list = shape(1,W_i * size)
			W_i : matrix = shape(n,m)
			n : previous layer size
			m : next layer size
	eta : double
			learning rate

	Y_hat : list = shape(60000,10)
			prediction
			10 : one of K
			60000 : number of samples
	
	y : list = shape(60000,10)
			10 : one of K
			60000 : number of samples

	A : the result of each layers

	-------

	Returns
	-------
	W : Weight of nnet which has been adjusted
			list = shape(1,W_i * size)
			W_i : matrix = shape(n,m)
			n : previous layer size
			m : next layer size
	-------
	'''
	# =====calculating delta
	n = len(W)
	delta_n = n
	delta = [1]*(n+1)
	for i in range(delta_n,0,-1):
		if i == delta_n:
			delta[i] = Y_hat - y
		else:
			delta[i] = np.multiply(np.dot(delta[i+1],W[i].T),np.multiply(A[i],1-A[i]))
	# =====updating W
	for i in range(n):
		W[i] = W[i] - eta/60000*delta_W
	return W
from base_function import Sigmoid,Linear,show_W,add_dummy_variable,data_T,OneofK
from nnet_init import W_init,Activation_function
from load_dataset import load_mnist
from nn_tr import nnet_calculation,Back_propagation
import numpy as np

X, y, X_test, y_test = load_mnist(only_binary = False)	
print "dataset reading complete,X.shape:",np.shape(X)[0],np.shape(X)[1]
print "dataset reading complete,y.shape:",np.shape(y)[0]
print "adding dummy variable to X.."
X = add_dummy_variable(X)
print "dummy variable adding complete,X.shape:",np.shape(X)[0],np.shape(X)[1]

print 'data tranposing..'
X,y = data_T(X,y)
print "data tranposing complete,X.shape:",np.shape(X)[0],np.shape(X)[1]
print "data tranposing complete,y.shape:",np.shape(y)[0]

print 'one of k coding to y'
y = OneofK(y)
print 'one of k coding complete,y.shape:',np.shape(y)

print "initial weight.."
W = W_init([100], X)
print "weight initialing complete"
print 'weight shape takes the form as follow'
print 'W_shape:',np.shape(W)

print 'initialing activation function..'
AF = Activation_function([Sigmoid,Sigmoid,Sigmoid])
print 'activation function initialing complete'

print 'nnet calculating..'
ERROR,Y_hat,A = nnet_calculation(W, AF, X, y)
print 'nnet calculating complete'
print 'ERROR',ERROR


time = 1000
t = 1

while t<time and ERROR>0.07:
	print '---------'
	print 'operator : ',t
	print 'Back propagation calculating..'
	W = Back_propagation(W,0.1,Y_hat,y,A)
	print 'Back propagation calculating complete'
	print 'nnet calculating..'
	ERROR,Y_hat,A = nnet_calculation(W, AF, X, y)
	print 'nnet calculating complete'
	print 'ERROR:',ERROR
	t = t + 1
	print '---------'

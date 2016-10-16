#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load_dataset import load_mnist
from FDA import FDA_train, FDA_test
import numpy as np

################################################################
#Loading data
#
#Variables
#---------
#X_train : list of lists, (n_samples, n_features) = (60000,784)
#          The features of training set
#y_train : array, shape(n_samples,) = (60000,)
#          The labels of training set range from 0 to 9
#X_test  : list of lists, (n_sampels, n_features) = (10000, 784)
#          The features of test set
#y_test  : array, shape(n_samples,) = (10000,)
################################################################
y_1 = 6
y_2 = 8

X_train, y_train, X_test, y_test = load_mnist(only_binary = True, y_1 = y_1, y_2 = y_2)#

#Separating X_train according to class label
X_train_0 = np.array([X_train[i] for i in xrange(len(y_train)) if y_train[i] == y_1])
X_train_1 = np.array([X_train[i] for i in xrange(len(y_train)) if y_train[i] == y_2])


#Training and Predicting
w_star, w_0 = FDA_train(X_train_0, X_train_1)
y_pred = FDA_test(X_test, w_star, w_0)

#one hot encoding of y_test(1-of-K code scheme)
y_0 = (np.mat(y_test) == y_1).astype(int).T
y_1 = (np.mat(y_test) == y_2).astype(int).T
y_test = np.hstack((y_0, y_1))

#Calculating the error rate
error_rate = abs(y_pred - y_test).sum() / (2. *len(y_test))
print ("The error rate is: %.4f"%error_rate)

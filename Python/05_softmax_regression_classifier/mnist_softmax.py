#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
from load_dataset import load_training, load_testing
from softmax import softmax_train, softmax_test
from basis_function import add_dummy_variable
from coding_scheme import one_hot_encoding
import numpy as np
import time


#=============================================================
#   Step 1: Load data

print "Loading training data..."
X_train, y_train = load_training()
print "Done"

#=============================================================
#   Step 2: 1-Of-K coding scheme

print "One hot encoding..."
T_train = one_hot_encoding(y_train)
del y_train
print "Done"

#=============================================================
#   Step 3: Training softmax
start = time.clock()
print "Training..."
W_optima = softmax_train(X_train, T_train, 2)
print "Done"
print u"Training stage takes :%.4f seconds"%(time.clock() - start)


#=============================================================
#   Step 4: Testing
del X_train, T_train
print "Testing..."
X_test, y_test = load_testing()
Phi_test= add_dummy_variable(X_test)
del X_test

m = np.shape(Phi_test)[1]
k = len(np.unique(y_test))
y_predict = softmax_test(W_optima, Phi_test, m, k)
print "Done"

#=============================================================
# Step 5: Calculating error rate

print "Calculating error rate..."
error_rate = len(np.nonzero(y_predict - y_test)) / len(y_test)

print "The error rate is %.4f"%error_rate

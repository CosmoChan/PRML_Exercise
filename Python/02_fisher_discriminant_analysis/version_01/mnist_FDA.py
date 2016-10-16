#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from load_dataset import load_mnist
from FDA import FDA_train, FDA_test
import numpy as np

X_train, y_train, X_test, y_test = load_mnist()

X_train_0 = np.array([X_train[i] for i in xrange(len(y_train)) if y_train[i] == 0])
X_train_1 = np.array([X_train[i] for i in xrange(len(y_train)) if y_train[i] == 1])

w_star, w_0 = FDA_train(X_train_0, X_train_1)
y_pred = FDA_test(X_test, w_star, w_0)

y_0 = (np.mat(y_test) == 0).astype(int).T
y_1 = (np.mat(y_test) == 1).astype(int).T
y_test = np.hstack((y_0, y_1))

error_rate = abs(y_pred - y_test).sum() / (2. *len(y_test))
print ("The error rate is: %.4f"%error_rate)

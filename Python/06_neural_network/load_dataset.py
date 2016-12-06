#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
from mnist import MNIST
import pandas as pd
import numpy as np


def load_mnist(only_binary = True, y_1 = 0, y_2 = 1):
    """Load mnist data set

    Load mnist dataset from PRML_Exercise/datasets folder
    mnist is python module,we can use "pip install python-mnist"
    to install this module easily.
    Here we use the MNIST function of mnist module

    Parameters
    ----------
    only_binary : bool
                  if only_binary is True, return the samples with
                  label y_1 or y_2, else return all the samples.
    y_1 : int
          The choosed label
    y_2 : int
          The choeesed label

    Returns
    -------
    X_train : array, shape(n_samples, n_features) = (60000, 784)
              The features of training set
    y_train : array, shape(n_sample, ) = (60000, )
              The labels of training set range from 0 9
    X_test : array, shape(n_samples, n_features) = (10000, 784)
             The features of test set
    y_test : array, shape(n_samples, ) = (10000, )
             The labels of test set
    """

    mndata = MNIST('../../datasets')

    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    if only_binary:
        X_train = X_train[(y_train == y_1) | (y_train == y_2), :]
        y_train = y_train[(y_train == y_1) | (y_train == y_2)]
        X_test = X_test[(y_test == y_1) | (y_test == y_2), :]
        y_test = y_test[(y_test == y_1) | (y_test == y_2)]

    X_train = normalization(X_train)
    X_test = normalization(X_test)

    return X_train, y_train, X_test, y_test

def normalization (X):
    n, m = X.shape
    mean = np.mean(X, axis = 1).reshape(n, 1)
    std = np.std(X, axis =1).reshape(n, 1)
    return (X - mean) / (std + 1e-11)

#!/usr/bin/env python
# encoding: utf-8

from mnist import MNIST
import pandas as pd
import numpy as np


def load_mnist(only_binary = True, y_1 = 0, y_2 = 1):
    """
    Load mnist dataset from PRML_Exercise/datasets folder
    mnist is python module,we can use "pip install python-mnist"
    to install this module easily.
    Here we use the MNIST function of mnist module to the load data
    """

    ################################################################
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

    mndata = MNIST('../../../datasets')
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()

    n_train,m_train = np.shape(X_train)
    n_test, m_test = np.shape(X_test)

    if only_binary:
        X_train = [X_train[i] for i in xrange(n_train) if (y_train[i] == y_1) or (y_train[i] == y_2)]
        y_train = [y_train[i] for i in xrange(n_train) if (y_train[i] == y_1) or (y_train[i] == y_2)]

        X_test = [X_test[i] for i in xrange(n_test) if (y_test[i] == y_1) or (y_test[i] == y_2)]
        y_test = [y_test[i] for i in xrange(n_test) if (y_test[i] == y_1) or (y_test[i] == y_2)]
    """
    n_train, m_train = np.shape(X_train)
    n_test, m_test = np.shape(X_test)

    X_train_mean = np.mean(X_train, axis = 1)
    X_test_mean = np.mean(X_test, axis = 1)
    train_mean_mat = np.array([X_train_mean, ] * m_train).T
    test_mean_mat = np.array([X_test_mean, ] * m_test).T

    X_train_std = np.mean(X_train, axis = 1)
    X_test_std = np.mean(X_test, axis = 1)
    train_std_mat = np.array([X_train_std, ] * m_train).T + 1e-6
    test_std_mat = np.array([X_test_std, ] * m_test).T + 1e-6

    X_train = (X_train - train_mean_mat) / train_std_mat
    X_test = (X_test - test_mean_mat) / test_std_mat
    """
    X_train = normalization(X_train)
    X_test = normalization(X_test)
    return X_train, y_train, X_test, y_test

def normalization (X):
    n, m = np.shape(X)
    mean = np.mean(X, axis = 1)
    mean_mat = np.array([mean, ] * m).T
    std = np.std(X, axis =1)
    std_mat = np.array([std, ] * m).T
    return (X - mean_mat) / std_mat

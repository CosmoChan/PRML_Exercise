#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
from mnist import MNIST
import pandas as pd
import numpy as np

def load_training(all_sample = True, target_label = [0,1,2,3,4,5,6,7,8,9]):
    """Load mnist training set

    Load mnist training set from PRML_Exercise/datasets folder.
    mnist is python module,we can use "pip install python-mnist"
    to install this module easily.
    Here we use the MNIST function of mnist module

    Parameters
    ----------
    all_sample : bool
                  If all_sample is True, return all the samples
                  in the training set
                  If all_sample is False, a list contains the target
                  labels should be given
    target_label : list
                    A list contains the choosen labels

    Returns
    -------
    X_test : array, shape(n_samples, n_features) = (10000, 784)
             The features of test set
    y_test : array, shape(n_samples, ) = (10000, )
             The labels of test set
    """
    mndata = MNIST('../../datasets')

    X_train, y_train = mndata.load_training()
    X_train, y_train = np.array(X_train), np.array(y_train)

    if not all_sample:
        index = []
        for each in y_train:
            if each in target_label:
                index.append(True)
            else:
                index.append(False)
        X_train = X_train[index, :]
        y_train = y_train[index]

    X_train = normalization(X_train)
    return X_train, y_train

def load_testing(all_sample = True, target_label = [0,1,2,3,4,5,6,7,8,9]):
    """Load mnist testing set

    Load mnist testing set from PRML_Exercise/datasets folder.
    mnist is python module,we can use "pip install python-mnist"
    to install this module easily.
    Here we use the MNIST function of mnist module

    Parameters
    ----------
    all_sample : bool
                  If all_sample is True, return all the samples
                  in the testing set
                  If all_sample is False, a list contains the target
                  labels should be given
    target_label : list
                    A list contains the choosen labels

    Returns
    -------
    X_train : array, shape(n_samples, n_features) = (60000, 784)
              The features of training set
    y_train : array, shape(n_sample, ) = (60000, )
              The labels of training set range from 0 9
    """
    mndata = MNIST('../../datasets')

    X_test, y_test = mndata.load_testing()
    X_test, y_test = np.array(X_test), np.array(y_test)

    if not all_sample:
        index = []
        for each in y_test:
            if each in target_label:
                index.append(True)
            else:
                index.append(False)
        X_test = X_test[index, :]
        y_test = y_test[index]

    X_test = normalization(X_test)
    return X_test, y_test

def normalization (X):
    n, m = X.shape
    mean = np.mean(X, axis = 1).reshape(n, 1)
    std = np.std(X, axis =1).reshape(n, 1)
    return (X - mean) / (std + 1e-11)

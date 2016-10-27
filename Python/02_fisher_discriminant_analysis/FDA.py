#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import numpy as np
from numpy.linalg import pinv
import time

def FDA_train(X_1, X_2):
    """Fisher's Discriminant Analysis training

    The FDA_train calculates the project direction w*
    and classification thredshold w_0 of FDA model.

    Parameters
    ----------
    X_1 : array-like, shape(n_samples, n_features)
          Training data of class 1
    X_2 : array-like, shape(n_samples, n_features)
          Training data of class 2

    Returns
    -------
    w_star : array-like, shape(n_features, )
         The best project direction w* under the
         Fisher's criterion
    w_0 : float
          The classification thredshold based on
          w_0 = -(1/2)(m_1_tilde + m_2_tilde)

    Notes
    -----
    To emphasize that FDA is a supervised learning method
    we'd better take X and y as parameters instead.
    However, in order to use the same symbols as
    teacher's slide, we still use X_1 and X_2
    """

    n_1,d_1 = np.shape(X_1)
    n_2, d_2 = np.shape(X_2)

    m_1 = np.mean(X_1, axis = 0)
    m_2 = np.mean(X_2, axis = 0)
    M_1 = np.array([m_1,]*n_1)
    M_2 = np.array([m_2,]*n_2)

    S_1 = ((X_1 - M_1).T).dot(X_1 - M_1)
    S_2 = ((X_2 - M_2).T).dot(X_2 - M_2)

    """
    for i in xrange(n_1):
        x, m = X_1[i].reshape(d_1, 1), m_1.reshape(d_1, 1)
        S_1 += (x - m).dot((x - m).T)


    S_2 = np.zeros((d_2, d_2))
    for i in xrange(n_2):
        x, m = X_2[i].reshape(d_2, 1), m_2.reshape(d_2, 1)
        S_2 += (x - m).dot((x - m).T)
    """
    S_w = S_1 + S_2

    #############################################
    #
    #Calculating the best project direction w^*
    #
    #Hint:
    #   you can use "pinv" to calculate the inverse
    #   matrix
    #   Use np.mat(object) to change object into
    #   matrix
    #
    #Note:
    #   m_1, m_2 are row vectors,please care about
    #   the dimension
    #
    #Your code goes here

    w_star = pinv(S_w).dot((m_1 - m_2).T)

    #############################################

    y_1 = X_1.dot(w_star)
    y_2 = X_2.dot(w_star)
    m_1_tilde = y_1.sum() / n_1
    m_2_tilde = y_2.sum() / n_2

    #Calculating the thredshold w_0
    ############################################
    #Calculating the thredshold w_0
    #
    #Your code goes here:


    w_0 = -(m_1_tilde + m_2_tilde) / 2
    ###########################################

    return w_star, w_0

def FDA_test(X_test, w_star, w_0):
    """Fisher Discriminant Analysis test

    The FDA_test projects the data points stored in X_test
    to the direction presented by w_star and compare
    the value y on the new axis with the
    specified threshold w_0, if y >= w_0 then the data
    point is classified into class 1, else class 2

    Parameters
    ----------
    X_test : array-like, shape(n_samples, n_features)
             The testing data
    w_star : array-like, shape(n_features, )
             The best project direction w* under the
             Fisher's criterion
    w_0    : float
             The classification thredshold based on
             w_0 = -(1/2)(m_1_tilde + m_2_tilde)

    Returns
    -------
    y_pred: array-like, shape(n_samples,k_classes)
             y_pred contains the class labels of each sample
             represented by 1-of-K format.(one hot encode)
    """

    y_proj = X_test.dot(w_star)[:, np.newaxis]

    #1-of-K code scheme
    y_1 = y_proj >= w_0
    y_2 = y_proj < w_0
    y_pred = np.hstack((y_1, y_2)).astype(int)

    return y_pred

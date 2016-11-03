#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import numpy as np
from numpy.linalg import pinv, inv, norm
from basis_function import add_dummy_variable
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


def softmax_train(X_train, T_train, lambda_, num_iteration = 10, tolerance = 1e-5):
    """Training softmax model
    Parameters
    ----------
    X_train : array-like, shape(n_samples, n_features)
            The features of training set
    T_train : array-like, shape(n_samples, n_labels)
            The labels of training set represented by 1-of-K coding scheme
    lambda_ : float
            The coefficient of regularization term
    num_iteration : int
            Maximum number of iterations
    tolerance : float
            The stop condition of iterations. If the norm of the first order derivative
            is less than tolerance, the loop will be stoped

    Returns
    -------
    w : array, shape(n_features, n_labels)
        The optimal parameters
    """

    Phi = add_dummy_variable(X_train)
    del X_train

    n, m = np.shape(Phi)
    n, k = np.shape(T_train)

    w = np.random.randn(m * k, 1) * 0.01

    plt.figure()
    print "Start Training..."

    for tao in xrange(num_iteration):
        print "Iteration: ", tao
        Y_predict = softmax_probability(w, Phi, m, k)
        E_w = softmax_cost(T_train, Y_predict)
        print "####################"
        print "#Cost: ", E_w
        print "####################"

        plt.plot(tao, E_w, 'r*')
        print "Calculating first order derivative"
        derivative = (first_order_derivative(T_train, Y_predict, Phi) + lambda_ * w)
        if norm(derivative) <= tolerance:
            print "The norm of the first order derivative is less than tolerance"
            break;
        print "Calculating second order derivative"
        H = Hessian(Y_predict, Phi, n, m, k) + np.eye(k*m) * lambda_

        print "Calculating inverse Hessian"
        w = w - inv(H).dot(derivative)
        print "The max(), min(), std() of parameters", w.max(), w.min(), w.std()
    plt.title('Lost Function vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('The Value of Lost Function')
    plt.show()
    w = w.reshape(k, m).T
    return w

def softmax_test(w_optima, Phi, m, k):
    """Testing softmax model
    Parameters
    ----------
    w_optima : array-like, shape(n_features, n_labels)
        The optimal parameters trained by softmax_train
    Phi : array-like, shape(n_samples, n_features)
        The features of testing set which was transformed by basis function
    m : int
        The number of features
    k : int
        The number of labels

    Returns
    -------
    Y_predict : array, shape(n_samples, 1)
        The predictions
    """
    product = Phi.dot(w_optima)
    probability = np.exp(product) / np.sum(np.exp(product), axis = 1)[:, np.newaxis]
    Y_predict = np.argmax(probability, axis = 1)
    return Y_predict


def softmax_probability(w_optima, Phi, m, k):
    """The probabilistic predictions
    Parameters
    ----------
    w_optima : array-like, shape(n_features, n_labels)
        The optimal parameters trained by softmax_train
    Phi : array-like, shape(n_samples, n_features)
        The features of testing set which was transformed by basis function
    m : int
        The number of features
    k : int
        The number of labels

    Returns
    -------
    probability : array, shape(n_samples, 1)
        The probabilistic predictions
    """
    w_optima = w_optima.reshape(k,m).T
    product = Phi.dot(w_optima)
    product = product - np.max(product, axis = 1)[:, np.newaxis]
    probability = np.exp(product) / np.sum(np.exp(product), axis = 1)[:,np.newaxis]
    return probability

def softmax_cost(T, Y):
    """The cost function of softmax
    Parameters
    ----------
    T : array-like, shape(n_samples, n_labels)
        The target variables encoded by 1-of-K scheme
    Y : array-like, shape(n_samples, n_labels)
        The probabilistic predictions encoded by 1-of-K scheme

    Returns
    -------
    total_cost : float
        The total cost of the n_samples
    """
    TY = T*Y
    predict_probability = TY[TY.nonzero()]
    cost_each_sample = - np.log(predict_probability + 1e-11)
    total_cost = cost_each_sample.sum()
    return total_cost

def first_order_derivative(T, Y, Phi):
    """First order derivative

    Calculating the negative gradient of cost function

    Parameters
    ----------
    T : array-like, shape(n_samples, n_labels)
        The target variables encoded by 1-of-K scheme
    Y : array-like, shape(n_samples, n_labels)
        The probabilistic predictions encoded by 1-of-K scheme
    Phi : array-like, shape(n_samples, n_features)
        The features of testing set which was transformed by basis function

    Returns
    -------
    first_order_derivative : array, shape(n_features*n_labels, 1)
        The negative gradient of cost function
    """
    Delta = Y - T
    first_order_derivative = Phi.T.dot(Delta)
    m, k = np.shape(first_order_derivative)
    first_order_derivative = first_order_derivative.T.reshape(m*k, 1)
    return first_order_derivative

def Hessian(Y, Phi, n, m, k):
    """Hessian matrix

    Constructing the hessian matrix

    Parameters
    ----------
    Y : array-like, shape(n_samples, n_labels)
        The probabilistic predictions encoded by 1-of-K scheme
    Phi : array-like, shape(n_samples, n_features)
        The features of testing set which was transformed by basis function
    n : int
        The number of samples
    m : int
        The number of features
    k : int
        The number of labels

    Returns
    -------
    H : array, shape(k*m, k*m)
        The hessian matrix
    """
    I = np.eye(k)
    H = np.zeros((k*m, k*m))
    for r in xrange(k):
        for c in xrange(k):
            H[r*m:(r+1)*m, c*m:(c+1)*m] = Phi.T.dot(Phi*(Y[:,c] * (I[c,r] - Y[:,r]))[:, np.newaxis])
    return H

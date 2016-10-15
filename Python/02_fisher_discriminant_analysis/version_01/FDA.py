#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def FDA_train(X_1, X_2):
    """Fisher's Discriminant Analysis training

    The FDA_train calculates the project direction w
    and classification thredshold w_0 of FDA model.

    Parameters
    ----------
    X_1 : array-like, shape(n_samples, n_features)
          Training data of class 1
    X_2 : array-like, shape(n_samples, n_features)
          Training data of class 2

    Notes
    -----
    To emphasize that FDA is a supervised learning method
    we'd better take X and y as parameters instead.
    However, in order to use the same symbols as
    teacher's slide, we still use X_1 and X_2
    """

    m_1 =

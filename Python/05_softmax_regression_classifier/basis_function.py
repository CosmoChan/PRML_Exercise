#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def add_dummy_variable(X):
    n, m = np.shape(X)
    dummy_variable = np.ones((n,1))
    Phi = np.hstack((dummy_variable, X))
    return Phi

#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def one_hot_encoding(y):
    """1-of-K coding scheme
    Parameters
    ----------
    y : array-like, shape(n_samples, 1)
        The labels.

    Returns
    -------
    T : array-like, shape(n_samples, n_categories)
    """
    labels = np.unique(y)
    n = len(y)
    column_list = []
    for label in labels:
        column_list.append((y == label).reshape(n,1))
    T = np.hstack(column_list).astype(int)
    return T

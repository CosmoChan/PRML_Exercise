# -*- coding: utf-8 -*-
"""
Least Squares Classifier

Created on Thu Sep 24 14:25:02 2016

@author: Voidsky
"""

from numpy import *
import operator
from os import listdir



def img2vect(filename):
	"""Loading a single sample to a vector.

	Each sample image's is a matrix whose pixels are 32*32,
	then we should transform this matirx to a 1*1024 vector.

	Args:
		filename: The file name of the training sample.

	Retruns:
		returnVect: The vector whose dimension is 1*1024.
	"""

	returnVect = zeros((1, 1024))
	fr = open(filename)

	# transform the matirx to a vector
	for i in range(32):
		lineStr = fr.readline()
		for j in range (32):
			returnVect[0, 32 * i + j] = int(lineStr[j])
	return returnVect



def getTrainingData():
	"""Import data of training set.

	From a file folder named "trainingDigits" import all of the
	data, using img2vect to import each data into a properties.
	matrix. And import the target matirx according to each filename.

	Args:
		None

	Retruns:
		X_tr: The matrix of the samples' properties.
		T_tr: The matrix about the samples' class.
	"""
	trainingFileList = listdir('../../datasets/trainingDigits')
	n = len(trainingFileList)
	X_tr = zeros((n, 1024));
	T_tr = zeros((n,10));

	# split the filename to obtain information
	for i in range(n):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		T_tr[i, classNumStr] = 1
		X_tr[i, :] = img2vect('../../datasets/trainingDigits/%s' % fileNameStr)
	return X_tr, T_tr



def lsc_tr(X_tr, T_tr):
	"""Calculate the parameters of the model.

	This a training function through which machine can calculate
	the best parameters of the linear model.

	Args:
		X_tr: The matrix of the training samples' properties.
		T_tr: The matrix about the training samples' class.

	Retruns:
		W: The estimates of the parameters.
	"""
	n = int(X_tr.shape[0])
	X_tr = c_[ones((n)), X_tr]
	W = dot(dot(linalg.pinv(dot(X_tr.T, X_tr)), X_tr.T), T_tr)
	return W



def getTestData():
	"""Import data of test set.

	From a file folder named "testDigits" import all of the
	data, using img2vect to import each data into a properties.
	matrix. And import the target matirx according to each filename.

	Args:
		None

	Retruns:
		X_te: The matrix of the test samples' properties.
		T_te: The matrix about the test samples' class.
	"""
	testFileList = listdir('../../datasets/testDigits')
	n = len(testFileList)
	X_te = zeros((n, 1024));
	T_te = zeros((n,10));

	# split the filename to obtain information
	for i in range(n):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		T_te[i, classNumStr] = 1
		X_te[i, :] = img2vect('../../datasets/trainingDigits/%s' % fileNameStr)
	return X_te, T_te



def lsc_te(X_te, W):
	"""Least Squares Cliassifier

	This a test function through which machine can estimate
	the class of the inputed test sample, according the  best
	parameters of the linear model. Pay attention that the case
	of a single sample and the case of

	Args:
		X_te: The matrix of the test samples' properties.
		W: The estimates of the parameters.

	Retruns:
		Y_te: The estimates of the test samples' class.
	"""
	n = int(X_te.shape[0])
	X_te = c_[ones((n)), X_te]
	if (n == 1): # A single sample
		Y_te = dot(W.T, X_te.T)
		k = where(Y_te == max(Y_te))[0][0]
		for i in range(10):
			if (i == k):
				Y_te[i] = 1
			else:
				Y_te[i] = 0
		Y_te = Y_te.T
	else: # Numbers of sample
		Y_te = dot(X_te, W)
		for i in range(n):
			k = where(Y_te[i, :] == max(Y_te[i, :]))[0][0]
			for j in range(10):
				if (j == k):
					Y_te[i, j] = 1
				else:
					Y_te[i, j] = 0
	return Y_te



def handwritingClassTest():
	"""Recognizing hand written class.

	Args:
		None

	Display:
		The error rate of the Least Squares Classifier.
	"""
	[X_tr, T_tr] = getTrainingData()
	W = lsc_tr(X_tr, T_tr)
	[X_te, T_te] = getTestData()
	testNum = len(X_te)
	Y_te = lsc_te(X_te, W)
	errorRate = sum(sum(abs(Y_te - T_te)))/float(2 * testNum)
	print "The error rate of the Least Squares Classifier is: %f" % errorRate

if __name__ == "__main__":
    handwritingClassTest()


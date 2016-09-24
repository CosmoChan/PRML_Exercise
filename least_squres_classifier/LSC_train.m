function [W] = LSC_train(X, T)
%In this helper function , we train our model
%i.e. calculating W
%Note that we add a dummy column in X
%   Parameters:
%       X: The images of training set
%       T: The labels of training set
%       W: Parameters of LSC.
X = [ones(1, size(X,2)); X]';

W = pinv(X' * X) * X' * T;

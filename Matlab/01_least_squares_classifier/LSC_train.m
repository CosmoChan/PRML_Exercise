function [W] = LSC_train(X, T)
%In this helper function , we will train our model
%i.e. calculating W
%Note that we add a dummy column in X
%   Parameters:
%       X: The images of training set
%       T: The labels of training set
%       W: Parameters of LSC.

%Calculating tilde X
X = [ones(1, size(X,2)); X]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Todo: Please compute W using the formula we have learned
%
%Your code goes here:

W = pinv(X' * X) * X' * T;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

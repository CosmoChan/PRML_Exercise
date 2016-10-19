clear;clc;
addpath ../../../datasets

%We use load_mnist to load image data of number 6 and 8.
[ train, test ] = load_mnist( true, 6, 8 );

train_X_6 = train.X(:,train.y == 6); 
train_X_8 = train.X(:,train.y == 8); 

[W, mean_6, mean_8 ] = FDA_tr( train_X_6, train_X_8 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% To do: please choose a formular we've learned to define the threshold w0
%
% Your code goes here:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To compute the labels of testing data.
y = FDA_te( test.X, W, w0 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% To do: please compute the accuracy of this classification case:
%
% Your code goes here:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the accuracy will be the same when uses different threshold
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
w0 = -(1/2)*( mean_6' * W + mean_8' * W );

w1 = - mean( [ train_X_6, train_X_8 ]' * W );

N1 = size( train_X_6, 2 );
N2 = size( train_X_8, 2 );
PC1 = N1 / ( N1 + N2 );
PC2 = N2 / ( N1 + N2 );
w2 = w0 - ( 1 / ( N1 + N2 - 2 ) )* log( PC1 / PC2 );
% you can only need to enter one of them in yourhomework
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To compute the labels of testing data.
y0 = FDA_te( test.X, W, w0 );
y1 = FDA_te( test.X, W, w1 );
y2 = FDA_te( test.X, W, w2 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% To do: please compute the accuracy of this classification case:
%
% Your code goes here:
accuracy0 = 1 - sum( abs( test.y - y0 ) )/( 2 * length( test.y ) )
accuracy1 = 1 - sum( abs( test.y - y1 ) )/( 2 * length( test.y ) )
accuracy2 = 1 - sum( abs( test.y - y2 ) )/( 2 * length( test.y ) )
% you can only need to enter one of them in yourhomework
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the accuracy will be the same when useing different threshold
% the accuracy is 98.24%
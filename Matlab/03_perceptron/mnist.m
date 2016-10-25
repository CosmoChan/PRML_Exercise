clear;clc;

% Please decide which two numbers you want to classify.
C1 = 6;
C2 = 8;

addpath ../../datasets

[ train, test ] = load_mnist( true, C1, C2 );

[ w ] = perceptron_tr( train.X, train.y, C1, C2 );

y = perceptron_te( w, test.X, C1, C2 );

accuracy = 1 - sum( abs( test.y - y ) )/( abs( C2 - C1 ) * length( test.y ) )
function [ y ] = FDA_te( test_X, W, w0 )
% This function is to classify the image data into number 6 or 8.
% Input: test_X is the dataset of the testing data.
%        W is the parameter of FDA and the output of FDA.
%        w0 is the threshold.

y = W' * test_X;
y( y > w0 ) = 6;
y( y < w0 ) = 8;

end
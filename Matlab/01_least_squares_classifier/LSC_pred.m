function Y_pred = LSC_pred(X_test, W)
%In this helper function , we test our model.
%The criterion is accuracy
%   Parameters:
%       X_test:     The images of test set
%       y_test:     The corresbonding labels of test set
%       W:              The parameters of LSC
%       T_test:     The corresbonding target vector(1-of-K representation of y_test)
%       Y_pred:     The predict values of test set 

%Calculating tilde X_te
X_test = [ones(1, size(X_test, 2)); X_test]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Todo: compute Y_pred using X_test and W
%
%Your code goes here:

Y_pred = X_test * W;

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Assign 1 for the maximum and 0 for others
for i = 1:size(Y_pred, 1)
    [~, idx] = max(Y_pred(i,:));
    Y_pred(i, idx) = 1;
    Y_pred(i, 3-idx) = 0;
end

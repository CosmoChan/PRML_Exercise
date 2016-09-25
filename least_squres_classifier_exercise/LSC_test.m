function accuracy = LSC_test(X_test, y_test, W)
%In this helper function , we test our model.
%The criterion is accuracy
%   Parameters:
%       X_test:     The images of test set
%       y_test:     The corresbonding labels of test set
%       W:              The parameters of LSC
%       n:               Number of testing samples
%       T_test:     The corresbonding target vector(1-of-K representation of y_test)
%       Y_pred:     The predict values of test set 

n = length(y_test);

%Predicting
X_test = [ones(1, size(X_test, 2)); X_test]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Todo: compute Y_pred using X_test and W
%
%Your code goes here:


%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Assign 1 for the maximum and 0 for others
for i = 1:size(Y_pred, 1)
    [~, idx] = max(Y_pred(i,:));
    Y_pred(i, idx) = 1;
    Y_pred(i, 3-idx) = 0;
end

%One hot encoding
T_test = oneOfK(y_test);

%The error rate is the same as accuracy
%I prefer accuracy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Todo: compute the error_rate and accuracy
%Hint: remember to divide by 2
%
%Your code goes here:


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

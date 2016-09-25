%If everything goes well, you will see that the accuracy is about 99.38
%Load the MNIST data for this exercise
%train.X and test.X will contain the training and testing images.
%Both matrix has size [n, m] where:
%   * m is the number of examples.
%   * n is the number of pixels.
%train.y and test.y will contain the corresponding labels(only binary digits in this case)

%We can change the last two parameters to specify the target digits
%we use 0 and 1 as default.
[train, test] = load_mnist(true, 0, 1);

%1-of-K presentation of so-called "One-Hot-Encoding"
T = oneOfK(train.y);

%Training the model:Calculating W
W = LSC_train(train.X, T);

%Testing the model
accuracy = LSC_test(test.X, test.y, W)

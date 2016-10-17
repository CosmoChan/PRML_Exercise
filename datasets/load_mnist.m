function [train, test] = load_mnist(binary_digits, y_1, y_2)
%The default case
if nargin == 0
    binary_digits = true;
    y_1 = 0;
    y_2 = 1;
end

  % Load the training data
  X=loadMNISTImages('train-images-idx3-ubyte');
  y=loadMNISTLabels('train-labels-idx1-ubyte')';

  if (binary_digits)
    % Take only the 0 and 1 digits
    X = [ X(:,y==y_1), X(:,y==y_2) ];
    y = [ y(y==y_1), y(y==y_2) ];
  end

  % Randomly shuffle the data
  I = randperm(length(y));
  y=y(I); % labels in range 1 to 10
  X=X(:,I);

  % Standardize the data so that each pixel will have roughly zero mean and unit variance.
  s=std(X,[],2);
  m=mean(X,2);
  X=bsxfun(@minus, X, m);
  X=bsxfun(@rdivide, X, s+.1);

  % Place these in the training set
  train.X = X;
  train.y = y;

  % Load the testing data
  X=loadMNISTImages('t10k-images-idx3-ubyte');
  y=loadMNISTLabels('t10k-labels-idx1-ubyte')';

  if (binary_digits)
    % Take only the 0 and 1 digits
    X = [ X(:,y==y_1), X(:,y==y_2) ];
    y = [ y(y==y_1), y(y==y_2) ];
  end

  % Randomly shuffle the data
  I = randperm(length(y));
  y=y(I); % labels in range 1 to 10
  X=X(:,I);

  % Standardize using the same mean and scale as the training data.
  X=bsxfun(@minus, X, m);
  X=bsxfun(@rdivide, X, s+.1);

  % Place these in the testing set
  test.X=X;
  test.y=y;


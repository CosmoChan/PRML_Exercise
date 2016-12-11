function [ train , test ] = load_MNIST_2d( is_std_normalize )

% Ĭ��is_std_normalizeΪ1����������Ƿ�Ϊ���ݽ��з����һ��
% �����ļ���һ���ļ�����һ��
% loadMNISTImages_2d.m,
% loadMNISTLabels.m, 
% train-images-idx3-ubyte, 
% train-labels-idx1-ubyte,
% t10k-images-idx3-ubyte,
% t10k-labels-idx1-ubyte
% ����ʾ�� [ train , test ] = load_MNIST_2d;


train.X = loadMNISTImages_2d('train-images-idx3-ubyte');
train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
test.X = loadMNISTImages_2d('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

Mean = mean( train.X ,3 );
train.X = bsxfun( @minus , train.X , Mean );
test.X = bsxfun( @minus , test.X , Mean );

if nargin == 0 || is_std_normalize
    Std = std( train.X , [] , 3 );    
    train.X = bsxfun( @rdivide , train.X , Std+eps );
    test.X = bsxfun( @rdivide , test.X , Std+eps );
end

digits = 0 : 9;

train.T = zeros( size( train_labels , 1 ) , 10 );
for k = 1 : 10
    train.T( train_labels==digits(k) , k ) = 1;
end

test.T = zeros( size( test_labels , 1 ) , 10 );
for k = 1 : 10
    test.T( test_labels==digits(k) , k ) = 1;
end

end
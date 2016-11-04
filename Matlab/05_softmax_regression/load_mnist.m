function [ train, test ] = load_mnist( digits )
% 描述：
%      load_mnist是数据加载函数，从原数据中加载指定数字的训练集合和测试集合
% 输入： 
%      digits 是一个向量，包含指定目标数字
% 输出：
%      train 是一个结构体，包含样本数据集合X和样本标签集合y。其中，X是矩阵，
%            X每一列都是一个样本数据。y是行向量，每个元素都是一个数值标签
%      test 是一个结构体，包含样本数据集合X和样本标签集合y。其中，X是矩阵，
%            X每一列都是一个样本数据。y是行向量，每个元素都是一个数值标签
% 代码如下：

addpath ../../../datasets

X_train_origin = loadMNISTImages('train-images-idx3-ubyte');
y_train_origin = loadMNISTLabels('train-labels-idx1-ubyte')';
X_test_origin = loadMNISTImages('t10k-images-idx3-ubyte');
y_test_origin = loadMNISTLabels('t10k-labels-idx1-ubyte')';

X_train=[];
y_train=[];
X_test=[];
y_test=[];
for i = 1 : length(digits)
    X_train = [ X_train , X_train_origin( : , y_train_origin == digits(i)) ];
    y_train = [ y_train , y_train_origin( y_train_origin == digits(i))     ];
    X_test  = [ X_test  , X_test_origin(  : , y_test_origin  == digits(i)) ];
    y_test  = [ y_test  , y_test_origin(  y_test_origin  == digits(i))     ];
end

I = randperm(length(y_train));
X_train = X_train( :, I);
y_train = y_train(I);

J = randperm(length(y_test));
X_test = X_test( :, J);
y_test = y_test(J);

train_s = std( X_train, [], 2 );
train_m = mean( X_train, 2 );

X_train = bsxfun( @minus, X_train, train_m );
X_train = bsxfun( @rdivide, X_train, train_s + 0.1 );
X_test = bsxfun( @minus, X_test, train_m );
X_test = bsxfun( @rdivide, X_test, train_s + 0.1 );

train.X = X_train;
train.y = y_train;
test.X = X_test;
test.y = y_test;
end




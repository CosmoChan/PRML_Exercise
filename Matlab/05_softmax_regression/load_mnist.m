function [ train, test ] = load_mnist( digits )
% ������
%      load_mnist�����ݼ��غ�������ԭ�����м���ָ�����ֵ�ѵ�����ϺͲ��Լ���
% ���룺 
%      digits ��һ������������ָ��Ŀ������
% �����
%      train ��һ���ṹ�壬�����������ݼ���X��������ǩ����y�����У�X�Ǿ���
%            Xÿһ�ж���һ���������ݡ�y����������ÿ��Ԫ�ض���һ����ֵ��ǩ
%      test ��һ���ṹ�壬�����������ݼ���X��������ǩ����y�����У�X�Ǿ���
%            Xÿһ�ж���һ���������ݡ�y����������ÿ��Ԫ�ض���һ����ֵ��ǩ
% �������£�

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




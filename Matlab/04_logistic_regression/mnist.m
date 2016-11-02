close all;
clear;
clc;
tic
%% 说明
% -mnist 是用Logistic Regression(LR)进行6和8手写数字识别的数据预处理、总体调度以及求出模型在测试集上的准确率的脚本

% 参数说明：
  %没有参数
  
%% 训练模型
% 导入训练数据集X_train，这是一个784*60000的矩阵，每一列是一个训练数据
X_train = loadMNISTImages('train-images-idx3-ubyte');

% 导入训练数据集的标签数据T_train，这是一个60000*1的向量
T_train = loadMNISTLabels('train-labels-idx1-ubyte');

%找出y=6和y=8的训练数据X_train(784*11769)和训练数据的标签T_train(1*11769)
X_train = X_train(:,T_train == 6 | T_train == 8);
T_train = T_train(T_train == 6 | T_train == 8)';

% 给训练数据加入偏置变量1，这是一个11769*785的矩阵，每一行是一个训练数据
X_train = [ones(1,size(X_train,2));X_train]';

% 将T_train标准化，使其等于6的变成1，否则等于0，这是一个11769*1的向量
T_train = SixToOne(T_train);

% 迭代次数的设置，如果迭代200次，模型的准确率为99.09%；可以自行设置迭代次数。
iteration = 200;

% 设置迭代停止的tolerance
tolerance = 0.000001;

% 训练模型得出模型的参数W，这是一个785*1的矩阵
W = logistic_tr(X_train,T_train,iteration,tolerance);

% 释放内存
clear X_train T_train;

%% 测试模型
% 导入测试数据X_test，这是一个784*10000的矩阵，每一列是一个训练数据
X_test = loadMNISTImages('t10k-images-idx3-ubyte');

% 导入测试数据的标签数据T_test，这是一个10000*1的向量
T_test = loadMNISTLabels('t10k-labels-idx1-ubyte');

%找出y=6和y=8的训练数据X_test(784*1932)和训练数据的标签T_test(1*1932)
X_test = X_test(:,T_test == 6 | T_test == 8);
T_test = T_test(T_test == 6 | T_test == 8)';

% 给测试数据加入偏置变量1，这是一个1932*785的矩阵，每一行是一个训练数据
X_test = [ones(1,size(X_test,2));X_test]';

% 将T_test标准化，这是一个1932*1的矩阵
T_test = SixToOne(T_test);

% 利用logistic_te.m得到测试数据的预测标签值T_pred，这是一个1932*1的矩阵
T_pred = logistic_te(W,X_test);

%释放内存
clear X_test;

%% 计算模型在测试数据上的准确率为99.39%
n = size(T_test,2);
accuracy =1- sum(abs(T_pred - T_test'))/(2 * n);

%释放内存
clear T_test,T_pred;
toc
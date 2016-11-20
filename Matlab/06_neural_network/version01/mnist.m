% mnist 是利用神经网络对手写数字进行分类
clear;
close all;
clc
tic
% 导入训练数据
train.X = loadMNISTImages('train-images-idx3-ubyte');
train.Y = loadMNISTLabels('train-labels-idx1-ubyte');

% 对Y进行OneOfK编码。train.X是784*60000的矩阵，train.Y是60000*1的向量
train.Y = OneOfK(train.Y);

% 定义神经网络学习速率alpha
alpha = 2;

% 定义神经网络迭代次数iteration
iteration = 1500;

% 训练神经网络模型
arg = nn_tr( train, alpha, iteration );

% 导入测试数据
test.X = loadMNISTImages('t10k-images-idx3-ubyte');
test.Y = loadMNISTLabels('t10k-labels-idx1-ubyte');

% 对Y进行OneOfK编码。test.X是一个784*10000矩阵,test.Y是一个10*10000的矩阵
test.Y = OneOfK(test.Y);

% 预测测试数据,10*10000
Y_pred = nn_te(test.X, arg);

% 计算准确率accuracy，准确率在95.7%左右，时间551s。
%如果想减少时间，可以适当提高学习速率alpha，并且减少迭代次数iteration
accuracy=1 - sum(sum(abs(test.Y - Y_pred)))/(2 * size(test.Y,2));
toc
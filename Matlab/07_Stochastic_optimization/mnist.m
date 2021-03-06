
% mnist 是利用三层神经网络对手写数字进行分类
clear;
close all;
clc
tic
%% 训练神经网络模型
% 导入训练数据
train.X = loadMNISTImages('train-images-idx3-ubyte');
train.Y = loadMNISTLabels('train-labels-idx1-ubyte');

% 对Y进行OneOfK编码。train.X是784*60000的矩阵，train.Y是60000*1的向量
train.Y = OneOfK(train.Y);

% 定义神经网络学习速率alpha
alpha = 125;

% 定义神经网络迭代次数iteration
iteration = 1000;

% 训练神经网络模型
arg = nn_tr( train, alpha, iteration );

%% 测试模型
% 导入测试数据plo
test.X = loadMNISTImages('t10k-images-idx3-ubyte');
test.Y = loadMNISTLabels('t10k-labels-idx1-ubyte');

% 对Y进行OneOfK编码。test.X是一个784*10000矩阵,test.Y是一个10*10000的矩阵
test.Y = OneOfK(test.Y);

% 预测测试数据,10*10000
Y_pred = nn_te(test.X, arg);

% 计算准确率accuracy，准确率在87%以上
accuracy=1 - sum(sum(abs(test.Y - Y_pred)))/(2 * size(test.Y,2));
toc

% 删除变量，释放内存。
clearvars -except accuracy arg
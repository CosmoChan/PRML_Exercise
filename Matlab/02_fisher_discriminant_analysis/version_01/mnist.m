%close all;
%clear;
%clc;

addpath ../../../datasets

%% 数据导入和初步处理
%导入训练数据，X_train是一个784*60000的矩阵，T_train是一个1*60000的向量
X_train=loadMNISTImages('train-images-idx3-ubyte');
T_train=loadMNISTLabels('train-labels-idx1-ubyte')';

%选择标签为6和8的数据，X_train是一个784*11769的矩阵，T_train是一个1*11769的向量
X_train = [ X_train(:,T_train==6), X_train(:,T_train==8) ];
T_train = [ T_train(T_train==6), T_train(T_train==8) ];
    
%对选择的训练数据加入偏置变量1，X_train是一个11769*784的矩阵
X_train = X_train';

%导入测试数据，X_test是一个784*10000的矩阵，T_test是一个10000*1的向量
X_test=loadMNISTImages('t10k-images-idx3-ubyte');
T_test=loadMNISTLabels('t10k-labels-idx1-ubyte');

%选择标签为6和8的数据，X_test是一个784*1932的矩阵，T_test是一个1*1932的向量
X_test = [ X_test(:,T_test==6), X_test(:,T_test==8) ];
T_test = [ T_test(T_test==6)', T_test(T_test==8)' ];

%对选择的测试数据加入偏置变量1，X_test是一个1932*784的矩阵
X_test = X_test';

%对测试数据进行乱序处理
Index = randperm(length(T_test));
T_test=T_test(Index);
X_test=X_test(Index,:);

%计算测试数据的数量
n = size(T_test,2);
%% 模型训练
%训练FDA_tr模型得到模型参数W和w
[W,w] = FDA_tr(X_train,T_train);

%% 模型检验准确率
%使用训练数据得到模型参数W和w计算测试数据的预测标签，得到的T_pred是一个1932*1的列向量
T_pred = FDA_te(X_test,W,w);

%计算测试数据在模型中的准确率。其中因为预测的标签值不是6就8，如果预测正确6-6=0,8-8=0，如果预测错误abs（6-8）=2，因此除以2*n
%----------------------your code here------------------------





%------------------------------------------------------------

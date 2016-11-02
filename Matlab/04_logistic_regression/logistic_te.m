function [ T_pred ] = logistic_te( W,X_test )
%logistic_te 这个函数是用logistic计算得出的参数W和测试数据求测试数据的标签预测值T_pred
%   参数说明：
%   W 训练模型得到的参数，这是一个785*1的向量
%   X_test 训练数据集，这是一个1932*785的矩阵
%   T_pred 训练数据的预测标签值，这是一个1932*1的向量
T = Sigmoid(X_test * W);
T_pred = zeros(size(T));
T_pred( T >= 0.5 ) = 1;
end


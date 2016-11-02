function [ W ] = logistic_tr( X_train,T_train,iteration,tolerance )
%logistic_tr 这个函数是利用牛顿法求解模型的参数W
%   参数说明
%   X_train 包含手写数字6和8的数据的训练集，这是一个11769*785的矩阵
%   T_train 训练集的标签向量，训练数据为6的标签为1，训练数据为8的标签为0，这是一个1*11769的向量
%   iteration 迭代的次数
%   tolerance 模型参数收敛的判别条件
%   W 训练得到的模型参数，这将是一个785*1的向量

% 计算X_train的大小
[~,n] = size(X_train);

% 随机初始化W
W =0.01 * randn(n,1);

eps = 1e-10;

% cost_old = 0;
for tao=1:iteration
    
    %计算类别归属概率的估计值，这是一个11769*1的向量
    y = Sigmoid( X_train * W );
    
    %计算损失函数
    cost = -sum( T_train' .* log( y - eps ) + ( 1 - T_train )' .* log( 1 - y + eps ));
%     if abs(cost - cost_old)<tolerance
%         break
%     end
%     
%     cost_old = cost;
    
    %计算对角矩阵R
    V = y .* (1 - y);
    R = diag( V );
    
    %更新权重
    W = W - pinv(X_train' * R * X_train ) * X_train' * ( y - T_train');
    tao
    cost
end
end


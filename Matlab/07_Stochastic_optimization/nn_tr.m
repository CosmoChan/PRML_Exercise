function arg = nn_tr( train, alpha, iteration )
%nn_tr 训练神经网络模型
%   train 训练数据
%   alpha 神经网络学习速率
%   iteration 神经网络最大迭代次数
%   W_1,W_2,b_1,b_2 神经网络的参数

[n,m] = size(train.X);

% 定义神经网络的参数
arg.W_1 = 0.05 * randn(100, 784);
arg.W_2 = 0.05 * randn(10, 100);
arg.b_1 = 0.05 * randn(100, 1);
arg.b_2 = 0.05 * randn(10, 1);

% 使用循环，逐步优化神经网络参数
for i=1:iteration
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %随机选择训练样本X，以及相应地标签Y







    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    a_1 = X;%784*60000
    
    % 计算隐藏层的输入加权和z_2
    z_2 = arg.W_1 * a_1 + arg.b_1;%100*60000
    % 计算隐藏层的激活值a_2
    a_2 = Sigmoid(z_2);%100*60000
    
    % 计算输出层的输入加权和z_3
    z_3 = arg.W_2 * a_2 + arg.b_2;%10*60000
    % 计算输出层的激活值a_3
    a_3 = softmax(z_3);%10*60000
    
    % 计算代价函数J
    J = 1/(2 * m) * sum(sum((a_3 - Y).*(a_3 - Y)));
    
    % 进行反向传播
    % 计算输出层的残差
    delta_3 = (a_3 - Y);% 10*60000
    % 计算隐藏层的残差
    delta_2 = (arg.W_2' * delta_3) .* a_2 .* (1 - a_2);% 100*60000
    
    % 计算梯度
    Delta_W_2 = delta_3 * a_2';%10*100
    Delta_b_2 = sum(delta_3,2);%10*1
    Delta_W_1 = delta_2 * a_1';%100*784
    Delta_b_1 = sum(delta_2,2);%100*1
    
    % 更新系数
    arg.W_1 = arg.W_1 - alpha * 1/m * Delta_W_1;
    arg.W_2 = arg.W_2 - alpha * 1/m * Delta_W_2;
    arg.b_1 = arg.b_1 - alpha * 1/m * Delta_b_1;
    arg.b_2 = arg.b_2 - alpha * 1/m * Delta_b_2;
    
end
end
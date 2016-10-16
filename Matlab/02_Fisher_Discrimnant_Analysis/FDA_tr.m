function [ W,w ] = FDA_tr( X_train,T_train )
% 函数FDA_tr是根据训练数据和标签计算投影方向W和判别阈值w
% X_train 和 T_train 是6、8的训练数据和标签

%分出标签为6和8的训练集
X_train_6 = X_train(T_train ==6,:);
X_train_8 = X_train(T_train ==8,:);

%计算每类的每个数据的各个参数的均值向量，这是一个1*785的行向量
m_six = mean(X_train_6);
m_eight = mean(X_train_8);

%计算每类的协方差矩阵，这是一个785*785的方阵
S_6 = (X_train_6-m_six)'*(X_train_6-m_six);
S_8 = (X_train_8-m_eight)'*(X_train_8-m_eight);

%计算类内离散度矩阵，这是一个785*785的方阵
S_w = S_6 + S_8;

%计算W，这是一个785*1的列向量
W = pinv(S_w)*(m_six-m_eight)';

%计算w
%pred_6 =X_train_6*W;
%pred_8 =X_train_8*W;
%m_6_pred = mean(pred_6);
%m_8_pred = mean(pred_8);
%w = -1/2*(m_6_pred+m_8_pred);
m_6_pred =m_six*W;
m_8_pred =m_eight*W;
w = -1/2*(m_6_pred+m_8_pred);
end


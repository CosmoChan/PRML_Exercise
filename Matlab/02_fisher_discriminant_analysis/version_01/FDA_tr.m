function [ W,w ] = FDA_tr( X_train,T_train )
% 函数FDA_tr是根据训练数据和标签计算投影方向W和判别阈值w
% X_train 和 T_train 是6、8的训练数据和标签

%分出标签为6和8的训练集
X_train_6 = X_train(T_train ==6,:);
X_train_8 = X_train(T_train ==8,:);

%计算每类的每个数据的各个参数的均值向量，这是一个1*784的行向量
m_six = mean(X_train_6);
m_eight = mean(X_train_8);

%计算每类的协方差矩阵S_6,S_8
%----------------------your code here------------------------

S_6 = (X_train_6-repmat(m_six,size(X_train_6,1),1))'*(X_train_6-m_six);
S_8 = (X_train_8-repmat(m_eight,size(X_train_8,1),1))'*(X_train_8-m_eight);




%------------------------------------------------------------

%计算类内离散度矩阵，这是一个784*784的方阵
S_w = S_6 + S_8;

%计算W
%----------------------your code here------------------------

W = pinv(S_w)*(m_six-m_eight)';



%------------------------------------------------------------


%计算w，采用课件上的第一种阈值
%----------------------your code here------------------------

m_6_pred =m_six*W;
m_8_pred =m_eight*W;
w = -1/2*(m_6_pred+m_8_pred);


%------------------------------------------------------------
end


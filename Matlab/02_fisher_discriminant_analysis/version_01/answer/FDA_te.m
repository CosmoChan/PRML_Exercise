function [ T_pred ] = FDA_te( X_test,W,w )
% -函数FDA_te是根据FDA_tr得出的W、w，利用测试数据得到预测的标签值
%X_test 是一个1932*785的矩阵，W是一个785*1的列向量

%计算测试数据的预测标签值，这是一个1932*1的列向量
T = X_test*W;

%将预测的标签值赋6和8，当将T<w的预测标签赋值6的时候，模型的准确率只有0.0212，显然不符合，所以将T>w的预测标签赋值6
T_pred = zeros(size(T,1),1);
T_pred(T>w)=6;
T_pred(T<w)=8;

end


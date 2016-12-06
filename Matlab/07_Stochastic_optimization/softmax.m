function [ a ] = softmax( Z )
%softmax 作为神经网络输出层的激活函数
%   Z 神经网络输出层的输入加权和
%   a 神经网络输出层的激活值
a = exp(Z) ./ (sum(exp(Z)));
end
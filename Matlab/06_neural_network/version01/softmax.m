function [ a ] = softmax( Z )
%softmax 使用softmax激活函数进行归一化处理
%   Z 神经网络输出层的输入值，包括偏置项,2*11769
%   a 神经网络最后一层的激活值
a = exp(Z) ./ (sum(exp(Z)));
end
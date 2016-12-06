function [ sigm ] = Sigmoid( a )
%Sigmoid 作为神经网络隐藏层的激活函数
%   a 神经网络隐藏层的输入加权和
%   sigm 神经网络隐藏层的激活值
sigm = 1 ./ (1 + exp(-a));
end
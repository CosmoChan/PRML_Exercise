function [ sigmoid ] = Sigmoid( a )
%Sigmoid 函数是使得a的值在0-1之间
%   参数说明
%   a ：一个向量或者标量或者矩阵
%   sigmoid ：标准化之后的a
sigmoid = 1 ./ (1 + exp(-a));

end


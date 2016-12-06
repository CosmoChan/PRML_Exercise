function [ T ] = OneOfK( Y )
%   OneOfK 将标签进行OneOfK编码
%   Y 待编码的标签向量
%   T 已经编码好的标签向量


% 声明标签矩阵T
T = zeros(10,size(Y,1));
% 如果Y(i) = 0，则T的第i列的第10的位置赋值为1，否则T的第i列的第Y(i)的位置赋值为1，其他为0
for i=1:length(Y)
    if Y(i) == 0
        T(10,i) = 1;
    else
        T(Y(i),i) = 1;
    end
end


function [ T ] = OneOfK( Y )
%   OneOfK 将标签进行OneOfK编码
%   Y 待编码的标签向量 60000*1
%   T 已经编码好的标签向量 10*60000


% 声明T，T是一个10*60000的矩阵 
T = zeros(10,size(Y,1));
for i=1:length(Y)
    if Y(i) == 0
        T(10,i) = 1;
    else
        T(Y(i),i) = 1;
    end
end


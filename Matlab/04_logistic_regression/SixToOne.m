function [ T_train ] = SixToOne( T )
%SixToOne SixToOne 将T中等于6的变成1，其他变成0
%   参数说明
%    T 输入的标签向量
T_train = zeros(size(T));
T_train(T == 6) = 1;
end


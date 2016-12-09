function cost = cross_entropy( Y , T )
% cross_entropy.m
%     用softmax作输出层激活函数时，用交叉熵作为代价函数
% 输入
%     Y 由输出层的softmax输出的矩阵，每行是一个样例
%     T 真实的分类标签T，每行是一个样例
% 输入
%     cost 为Y,T样本的交叉熵

[ n , ~ ] = size( Y ); 

cost = - sum( sum( T .* log( Y ) ) ) / n;

end
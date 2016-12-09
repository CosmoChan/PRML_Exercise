function cost = mean_square_error( Y , T )
% mean_square_error.m
%     对于回归问题，使用均方差作为代价函数
% 输入
%     Y 由输出层的所输出的矩阵，每行是一个样例
%     T 真实的分类标签T，每行是一个样例
% 输入
%     cost 输出 1/2均方误差

[ N , ~ ] = size( Y );

cost = sum( sum( ( ( Y - T ) .^ 2 ) ) ) / ( 2 * N );

end
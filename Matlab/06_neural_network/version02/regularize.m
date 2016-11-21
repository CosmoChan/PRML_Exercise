function T = regularize( Y )
%将Y中每行的最大的元素变为1，其余变为0

[ ~ , position ] = max( Y ,[] , 2 );    %获取Y每行最大的最大元索引

T = zeros( size( Y ) );                 %构造0矩阵

for i = 1 : size( Y , 1 )

    T( i , position( i ) ) = 1;         %将Y每行的最大值赋为1

end

end
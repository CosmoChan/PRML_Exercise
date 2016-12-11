function [ sample , origin_position ] = max_sampling( X , sampling_size )
% 最大池化，返回池化结果以及最大值位置
% 示例 [ max_value , max_position ] = max_sampling( ones( 1000 , 1000 , 123 ) , 2 )

[ ROWS , COLUMNS , N ] = size( X );

rows = ROWS / sampling_size;

columns = COLUMNS / sampling_size;

s = rows * columns;

sample = zeros( rows * columns , sampling_size^2 , N );

k = 0;

for r = 1 : sampling_size
    for c = 1 : sampling_size
        k = k + 1;
        sample( : , k , : ) = reshape( X( r:sampling_size:end , c:sampling_size:end , : ) , s , 1 , N );
    end
end

[ sample , position ] = max( sample , [] , 2 );

sample = reshape( sample , rows , columns , N );

P = zeros( s , sampling_size^2 , N );

for i = 1 : s
    P( i , position( i , 1 , : ) , : ) = 1;
end

origin_position = zeros(size(X));

k = 0;
for r = 1 : sampling_size
    for c = 1 :sampling_size
        k = k + 1;
        origin_position( r:sampling_size:end , c:sampling_size:end , : ) = reshape( P( : , k , : ) , rows , columns , N );
    end
end

%同时返回一个具有size(X)的 0 1矩阵，记录位置

end


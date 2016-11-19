function Y = softmax_hypothesis_function( X , W , is_column )
%  softmax_hypothesis_function： 
%     softmax函数将估计每个样本属于K类中每一类的概率
% 输入：
%     X 是输入数据矩阵，若X是n行d列，则表示有n个样本，每个样本有d个维数
%     W 是softmax模型的参数，对于K分类问题，W的输入有如下要求：
%             当输入参数is_column缺省或者为0的时候，W是d行K列的矩阵
%             当输入参数is_column为非零值，即真值得时候，W是d*K行1列的向量
% 输出：
%     Y 为n行K列矩阵，其元素Y(i,j)表示X中第i行的样本属于第j类的概率

%如果W是d*K行1列参数向量，那么将它转换为d行K列的矩阵
if nargin == 3 && is_column ==1
    
    %获取输入向量的维数
    [ ~ , d ] = size( X );
    
    %获取分类数量K
    K = length( W ) / d;
    
    %将d*K行1列的向量转换成d行K列的矩阵
    W = reshape( W , d , K );
else   
    %如果输入的W是d行K列矩阵，那么直接获取分类数量K
    [ ~ , K ] = size( W ); 
end

Y = exp( X * W );

%Y的每一行中每一个元素都除以该行的和，从而概率归一化
Y = bsxfun( @rdivide , Y , sum( Y , 2 ) );

%S = sum( Y , 2 );
%for i = 1 : K    
%    Y(:,i) = Y(:,i) ./ S;    
%end

end


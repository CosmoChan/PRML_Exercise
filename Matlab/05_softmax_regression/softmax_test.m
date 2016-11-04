function T = softmax_test( X , W )
% softmax_test：
%     测试函数，用给定的模型参数矩阵W，对给定的测试集合进行分类估计
% 输入：
%     X 输入集合的矩阵，为n行d列矩阵，每行是一个输入向量，输入向量有d维
%     W 是模型参数矩阵，为d行K列矩阵，每列是一个相应类型的模型参数向量
% 输出：
%     T 是标签矩阵，为n行K列矩阵，每行是一个标签

%用softmax函数估计每个样本属于K类中每一类的概率

Y = softmax_hypothesis_function( X , W );

%从Y每行中选出最大概率的列标号
[ ~ , Position] = max( Y , [] , 2 );

%获取样本数n
[ n , ~ ] = size( X );

%获取分类数K
[ ~ , K ] = size( W );

%构造n行K列空矩阵
T = zeros( n , K );

%对于n个样本，将其分为概率最大的一类
for i = 1 : n  
    T( i , Position( i ) ) = 1;    
end

end
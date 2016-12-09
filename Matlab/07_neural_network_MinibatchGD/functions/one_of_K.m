function labels = one_of_K( elements , targets )
% one_of_K：
%     依据targets向量，将elemets中每个元素编码成"one-of-K"向量形式的标签
% 输入：
%     elements 是n维向量，其元素可以是数值，可以是字符，n可以为1
%     targets 是K维向量，包含所有分类目标
% 输出
%     vector_labels K行n列向量，其第i列是elements中第i个元素的编码
% 示例：
%     输入：
%               >>one_of_K( [ 2 1 2 3 2 1 0 2 3 ] , [ 0 1 2 3 ])
%     输出结果：
%               0  0  0  0  0  0  1  0  0
%               0  1  0  0  0  1  0  0  0
%               1  0  1  0  1  0  0  1  0
%               0  0  0  1  0  0  0  0  1
%      
%     输入：
%               >>one_of_K( [ 'b' , 'd' ] , [ 'a' , 'b' , 'c' ] )
%     输出结果：
%               0  0
%               1  0
%               0  0

%获取标签数量
n = length( elements );

%获取目标标签的种类数量
m = length( targets );

%构造空的标签矩阵用于存放标签向量
labels = zeros( m , n );

%将elements中的每一个元素与向量targets匹配，将索引作为编码存入矩阵
for i = 1 : n
    j = find( targets == elements(i) );
    labels( j , i ) = 1;
end

end

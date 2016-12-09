function Z = softmax( a )

%要求每行是一个输入向量

Z = exp( a );

%Y的每一行中每一个元素都除以该行的和，从而概率归一化
Z = bsxfun( @rdivide , Z , sum( Z , 2 ) );

end


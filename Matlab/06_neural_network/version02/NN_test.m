function Y = NN_test( X , Wb , activations )

W = Wb{ 1 };                                  %从系数集合Wb中取出W集合

b = Wb{ 2 };                                  %从系数集合Wb中取出b集合

activations = [ {[]} , activations ];         %输入层没有激活函数，加入空元胞占位

L = length( W );                              %获取不含输出层的层数L，则共有L+1层

A = X;                                        %取第一层的单元输出矩阵 A=X

for l = 1 : L                                 %前向传播
    
    Z = bsxfun( @plus , A * W{ l } , b{ l }); %将 l 层的单元输出矩阵传播到第 l+1 层
    
    A = activations{ l + 1 }( Z );            %用第 l+1 层的激活函数处理 第 l+1 层的单元输入矩阵
        
end

Y = A;                                        %最后一层的单元输出矩阵就是网络的输出结果

end


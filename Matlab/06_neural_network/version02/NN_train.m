function Wb = NN_train( X , T , config , activations , derivatives , max_iterations , eta  )
% 神经网络训练程序
%
% 输入：
%    X 是样本的输入矩阵，每行是一个样例
%    T 是样本的输出矩阵，每行是一个样例
%    config 是一个向量，向量的维数控制神经网络的层数L，其各元素大小控制各层的但单元数量
%    activations 是一个元胞数组，包含 L-1 个 激活函数句柄，依此对应第2,3,...,L层的激活函数
%    dervatives 是一个元胞数组，包含 L-1 个与激活函数相应的 导函数的句柄
%    max_iterations 是一个正整数，是最大迭代次数，本程序用最大迭代次数来控制迭代停止
%    eta 是一个非负数，为学习速率。若设置太大，则损失函数将发生震荡，难以收敛，若太小，则学习效率低
% 输出：
%    Wb 参数集合，包含各层的W集合和b集合

L = length( config );                 %取从输入层到输出层的总层数
[ N , ~ ] = size( X );                %取样本大小
eta = eta / N;                        %将梯度下降步长除以样本大小N

activations = [ {[]} , activations ]; %由于输入层不设置激活函数，前面添加空元胞以占位
derivatives = [ {[]} , derivatives ]; %同上

A = cell( L , 1 );                    %存放各层的单元输入矩阵，其中第一个仅用来占位
Z = cell( L , 1 );                    %存放各层的单元激活输出矩阵
Delta = cell( L , 1 );                %存放各层的残差矩阵，其中第一个仅用来占位
errors = zeros( max_iterations , 1 ); %设置数组errors，用于记录每次的迭代的损失函数值
                                      
W = cell( L - 1 , 1 );                %存放前L-1层的系数矩阵 W
b = cell( L - 1 , 1 );                %存放前L-1层的偏置系数向量 b
for l = 1 : L-1                       %初始化各层参数 W b
    W{ l } = 0.1 * randn( config( l ) , config( l + 1 ) );  
    b{ l } = 0.1 * randn( 1 , config( l + 1 ));
end

Z{ 1 } = X;                           %设置第一层单元的输出矩阵 Z 为 X

for iterations = 1 : max_iterations   %开始迭代，完成指定次数之后跳出循环

    iterations                                %输出迭代次数

    for l = 1 : L-1                           %从1到L-1层，进行前向传播
        
        A{ l + 1 } = bsxfun( @plus , Z{ l } * W{ l } , b{ l });
        
        Z{ l + 1 } = activations{ l + 1 }( A{ l + 1 } );
        
    end                               %网络输出值为 Y := Z{ L }
    
    Delta{ L } = derivatives{ L }( A{ L } ) .* ( Z{ L } - T );%计算最后一层的残差矩阵
    
    errors( iterations ) = sum(sum(( Z{ L } - T ).^2 )) / N;  %计算损失函数
    
    errors( iterations )                      %输出损失函数
    
    for l = L-1 : -1 : 1                      %从第L-1到第1层，进行误差反向传播
        
        Gradient_W = Z{ l }' * Delta{ l + 1 };         %计算第l层系数矩阵 W 的梯度
        
        Gradient_b = ones( 1 , N ) * Delta{ l + 1 };   %计算第l层系数向量 b 的梯度
        
        W{ l } = W{ l } - eta * Gradient_W;            %对系数矩阵 W 进行梯度下降
        
        b{ l } = b{ l } - eta * Gradient_b;            %对偏置系数向量 b 进行梯度下降

        if l ~= 1                              %如果l不是第一层，那么计算该层的残差矩阵
            
            Delta{ l } = derivatives{ l }( A{ l } ) .* ( Delta{ l + 1 } * W{ l }' );
            
        end
        
    end
    
end

plot( errors , '.b' )                  %绘制误差函数随着iterations变化的函数图像

Wb = { W , b };

end
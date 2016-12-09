function Wb = NN_train( X , T , struct , activations , derivatives , cost_function , minibatch_size , max_epochs , eta  )
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

L = length( struct );                 %取从输入层到输出层的总层数
[ N , ~ ] = size( X );                %取样本大小
if minibatch_size > N
    error('minibatch不可大于训练样本总量')
end
max_iterations = ceil( max_epochs * N / minibatch_size );%最大迭代次数等于 整个训练次数上限 * 样本总数 / 小批量数

activations = [ {[]} , activations ]; %由于输入层不设置激活函数，前面添加空元胞以占位
derivatives = [ {[]} , derivatives ]; %同上

Z = cell( L , 1 );                    %存放各层的单元输入矩阵，其中第一个仅用来占位
A = cell( L , 1 );                    %存放各层的单元激活输出矩阵
Delta = cell( L , 1 );                %存放各层的残差矩阵，其中第一个仅用来占位
errors = zeros( max_iterations , 1 ); %设置数组errors，用于记录每次的迭代的损失函数值
                                      
W = cell( L - 1 , 1 );                %存放前L-1层的系数矩阵 W
b = cell( L - 1 , 1 );                %存放前L-1层的偏置系数向量 b
for l = 1 : L-1                       %初始化各层参数 W b
    r = sqrt( 6 / ( struct( l ) + struct( l + 1 ) ) );          % xavier经验公式
    W{ l } = 2 * r * rand( struct( l ) , struct( l + 1 ) ) - r;  
    b{ l } = zeros( 1 , struct( l + 1 ) );
end

batch_head = 1;                         %minibatch的首样本索引
iterations = 0;                         %小批量的迭代次数
go_on = 1;                              %go_on，预留的迭代停止指标的，可以尝试添加迭代停止条件 
while iterations < max_iterations && go_on
    iterations = iterations + 1;
    
    batch_tail = batch_head + minibatch_size - 1;  %从样本的X和T中，按顺序循环抽取minibatch_size个训练数据
    if batch_tail <= N
        A{ 1 } = X( batch_head : batch_tail , : );
        batch_T = T( batch_head : batch_tail , : );
    else
        batch_tail = batch_tail - N;
        A{ 1 } = [ X( batch_head : end , : ) ; X( 1 : batch_tail , : ) ];
        batch_T = [ T( batch_head : end , : ) ; T( 1 : batch_tail , : ) ];
    end
    batch_head = mod( batch_tail , N ) + 1;

    for l = 1 : L-1                           %从1到L-1层，进行前向传播
        
        Z{ l + 1 } = bsxfun( @plus , A{ l } * W{ l } , b{ l } );
        
        A{ l + 1 } = activations{ l + 1 }( Z{ l + 1 } );
        
    end                               %网络输出值为 Y := A{ L }
    
    Delta{ L } = derivatives{ L }( Z{ L } ) .* ( A{ L } - batch_T );%计算最后一层的残差矩阵
    
    errors( iterations ) = cost_function( A{ L } , batch_T );  %计算损失函数
    
    fprintf('epochs: %.2f, cost: %f\n', iterations * minibatch_size / N , errors( iterations ) );
    
    for l = L-1 : -1 : 1                      %从第L-1到第1层，进行误差反向传播
        
        Gradient_W = A{ l }' * Delta{ l + 1 } / minibatch_size;%计算第l层系数矩阵 W 的梯度
        
        Gradient_b = sum( Delta{ l + 1 } ) / minibatch_size;   %计算第l层系数向量 b 的梯度
        
        W{ l } = W{ l } - eta * Gradient_W;            %对系数矩阵 W 进行梯度下降
        
        b{ l } = b{ l } - eta * Gradient_b;            %对偏置系数向量 b 进行梯度下降

        if l ~= 1                              %如果l不是第一层，那么计算该层的残差矩阵
            
            Delta{ l } = derivatives{ l }( Z{ l } ) .* ( Delta{ l + 1 } * W{ l }' );
            
        end
        
    end
    
end

plot( errors , '.b' , 'MarkerSize',3 )         %绘制误差函数随着iterations变化的函数图像
xlabel('iterations')
ylabel('cost')

Wb = { W , b };

end
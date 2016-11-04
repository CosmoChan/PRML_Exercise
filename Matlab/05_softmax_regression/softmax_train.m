function [ W , iterations ] = softmax_train( X , T , lambda )
% softmax_train：
%     输入训练数据的输入向量集合矩阵X和标签集合矩阵T，以及正则化系数lambda
%     利用Newton-Raphson迭代求出softmax模型的参数矩阵W，并输出迭代次数
% 输入：
%     X 是n行d列矩阵，每行是一个输入向量，每个输入向量有d维
%     T 是n行K列矩阵，每行是标签，每行的K个元素中有一个为1，其余为0
%     lambda 是正则化系数，防止参数W的二范数过大
% 输出：
%     W 是d行K列矩阵，K列中每列都是相应类别的参数向量
%     iterations 迭代次数

%获取输入向量的维数
[ ~ , d ] = size( X );

%获取分类个数
[ ~ , K ] = size( T );

%设置参数W的迭代初值，为接近0的d*K行1列的向量
W = rand( d * K , 1 )*0.01;%%

%迭代步骤，最大迭代次数为15
for iterations = 1 : 15

    %用softmax函数估计每个样本属于K类中每一类的概率，模型参数W是d*K行1列的向量
    Y = softmax_hypothesis_function( X , W , 1 );

    %将概率矩阵和分类标签矩阵作差
    Delta = Y - T;

    %计算K类中，逐类的梯度向量，拼接成一个d*k行的梯度列向量
    Gradient = zeros( d * K , 1 );
    
    for j = 1 : K
        %计算第j类的参数w_j
        Gradient( 1 + (j-1)*d : j*d ) = sum( bsxfun( @times , X' , Delta( : , j )') , 2 );       
    end
    
    %梯度向量中加入正则化项
    Gradient = Gradient + lambda * W;
    
    %构造Hessian矩阵的K×K个子块，并拼接成d*K行d*K列的Hessian矩阵
    Hessian = zeros( K * d );
    for k = 1 : K   
        for j = 1 : K         
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %代码填在下面空白处
            

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %将第k,j个子块存入Hessian的相应位置
            Hessian( 1+(k-1)*d : k*d , 1+(j-1)*d : j*d ) =  Sub_hessian;
            
        end
    end
    
    %Hessian矩阵中加入正则化项
    Hessian = Hessian + lambda * eye( K * d );
    
    %更新模型参数列向量W。
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %代码下面空白处，提示：求逆时用A\B代替inv(A)*B

    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %迭代终止判断，如梯度Gradient的二范数接近0则停止迭代
    if norm( Gradient ) < 200
        break;        
    end
    
end

%将模型参数列向量W转换成d行K列的矩阵，作为结果输出
W = reshape( W , d , K );

end


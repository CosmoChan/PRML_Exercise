function [ GAMMA , MU , SIGMA , PI , LogLikehood ] = Gaussian_Mixture_EM( X , K , threshold , max_iterations )
% Gaussian_Mixture_EM.m
%     高斯混合模型EM算法实现程序，采用K-means算法来初始化参数初值 wuweizhen version
% 输入
%     X 是数据样本矩阵，每行是一个观测样例
%     K 是分类团簇数
%     threshold 迭代停止条件阈值，当对数似然函数变化量小于threshold时停止迭代
%     max_iterations 最大迭代次数
% 输出
%     GAMMA 每个观测样例属于各个类别的后验概率，每行是一个样例，每列是一个类别
%     MU  均值矩阵，共有K行，每行是一个高斯分布的均值向量
%     SIGMA 协方差的三维数组，其每一页是一个高斯分布的协方差矩阵
%     PI  先验概率列向量，第k个元素是数据属于k类别的先验概率
%     LogLikehood 对数似然函数

[ N , ~ ] = size( X );                                                          %获取样本大小N

[ MU , SIGMA , PI ] = GM_initialization( X , K , threshold );                   %用K-means算法初始化参数

P = zeros( N , K );                                                             %创建一个矩阵，P(n,k)表示第n个观测属于类别k的概率

LogLikehood_old = -inf;                                                         %初始化对数似然函数

for iterations = 1 : max_iterations

    for n = 1 : N
        
        for k = 1 : K                                                           
            %-------------------------------------------------------
            %计算第n个观测值属于类别k的概率
            %利用第k类的均值向量MU(k,:)、协方差矩阵SIGMA(:,:,k)，计算第k个高斯分布在x_n处的概率密度值，然后乘上第k类的先验概率
            %可以使用matlab自带的多元高斯分布概率密度函数来计算N(x_n|MU_k,SIGMA_k)，请help mvnpdf。也可以根据公式自己编写
            
            P( n , k ) =          
            
            %-------------------------------------------------------                      
        end
        
    end

    LogLikehood = sum( log( sum( P , 2 ) ) , 1 );                               %计算对数似然函数
    
    fprintf('K:%i, iter: %i/%i, log likehood: %f\n', K , iterations , max_iterations , LogLikehood )%在屏幕上显示每次迭代的信息
    
    if LogLikehood - LogLikehood_old < threshold                                %如果对数似然函数变化量小阈值，停止迭代

        break;
        
    end
    
    LogLikehood_old = LogLikehood;                                              %更新旧的对数自然函数，用于比较前后两次变化量
    %-------------------------------------------------------
    %计算责任矩阵GAMMA，其中GAMMA(n,p)表示观测值x_n属于第k类的后验概率
    %计算N_k
    %计算各个类别的先验概率PI
    %更新各类的均值矩阵MU，每行是一个类别，每列是一个维度
    %更新各类的均值矩阵MU，其中每行是一个类别，每列是一个维度
    %这5个计算都可以尝试使用一行代码完成（共5行）
    
    
    
    
    
    
    
    
    
    
    
    
    
    %-------------------------------------------------------
    
    for k = 1 : K                                                               %计算第k类的协方差矩阵
        
        delta = bsxfun( @minus , X , MU( k , : ) );                             %首先计算各个观测与第k类均值向量的差矩阵
        
        SIGMA( : , : , k ) = bsxfun( @times , delta' , GAMMA( : , k )' ) * delta / N_k( k );
                          %等价于 delta' * diag( GAMMA( : , k )' ) * delta / %N_k( k )        
    end

end

end
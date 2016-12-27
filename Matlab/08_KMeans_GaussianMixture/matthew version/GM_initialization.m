function [ MU , SIGMA , PI ] = GM_initialization( X , K , max_iterations )
% GM_initialization.m
%     混合高斯模型EM算法的初始化函数，利用K-means算法来生成初始化参数
%     该函数将被Gaussian_Mixture_EM函数自动调用
% 输入
%     X 是数据样本矩阵，每行是一个观测样例
%     K 是分类团簇数
%     max_iterations K-means运行的最大迭代次数
% 输出
%     MU    初始化的均值矩阵，共有K行，每行是一个高斯分布的均值向量
%     SIGMA 初始化的协方差三维数组，其每一页是一个高斯分布的协方差矩阵
%     PI    初始化的先验概率列向量，第k个元素是数据属于k类别的先验概率

[ ~ , d ] = size( X );

[ labels , MU , ~ ] = K_means( X , K , max_iterations );

SIGMA = zeros( d , d , K );

PI = zeros( K , 1 );

for k = 1 : K
    
    SIGMA( : , : , k ) = cov( X( labels == k , : ) );
    
    PI( k ) = mean( labels == k );
    
end

end
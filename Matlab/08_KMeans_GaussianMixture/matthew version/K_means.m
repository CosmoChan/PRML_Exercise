function [ labels , means , cost ] = K_means( X , K , max_iterations )
% K_means.m
%     K-means算法实现程序 wuweizhen version
% 输入
%     X 是数据样本矩阵，每行是一个观测样例
%     K 是分类团簇数
%     max_iterations 最大迭代次数
% 输出
%     labels 标签列向量，其每个元素表示X中相应行所属类别
%     means  均值矩阵，共有K行，每行是一个团簇的均值向量，即团簇的中心点
%     cost   损失函数

[ N , d ] = size( X );                                      %获取样本数N和维数d

labels = ceil( K * rand( N , 1 ) );                         %随机生成N个的标签

means = zeros( K , d );                                     %创建矩阵，用于存放K个团簇均值向量

for k = 1 : K
    means( k , : ) = mean( X( labels == k , : ) , 1 );      %计算各个均值向量
end

changed = 1;                                                %设置一个记录聚类是否发生改变的变量
iterations = 0;                                             %迭代计数器
while changed == 1 && iterations < max_iterations           %当迭代达到上限或者聚类不改变时，停止迭代
    
    changed = 0;
    
    for n = 1 : N                                           %考察第n个点
        %----------------------------------------------------
        %计算x_n到K个类别的均值向量的欧拉距离，求最小者，将x_n划分至该类
        %如果x_n类别发生变动，那么更新各类的均值向量，并且将changed修改为1
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        %----------------------------------------------------
    end
end

cost = 0;                                                   %设置变量记录损失函数

for k = 1 : K                                               %将每个团簇上的损失函数值加起来
                                                            %计算第k个团簇的各个点到团簇中心的差矩阵
    delta = bsxfun( @minus , X( labels == k , : ) , means( k , : ) );
    
    distance = sum( delta.^2 , 2 );                         %计算第k个团簇的各个点到团簇中心的欧拉距离
    
    cost = cost + sum( distance );                          %将向量求和，得到第k类的损失函数，然后累加到总的损失函数中
    
end

end
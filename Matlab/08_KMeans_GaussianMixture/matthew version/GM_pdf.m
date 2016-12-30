function P = GM_pdf( X , MU , SIGMA , PI )
% GM_pdf.m
%     高斯混合模型的概率密度函数，对于给定的模型参数，求模型在一个或多个点上的概率密度值
% 输入
%     X     Nxd矩阵，每行是一个点，每列是一个维度
%     MU    Kxd矩阵，每行是一个类别的均值向量，每列是一个维度
%     SIMGA dxdxK三维数组，每页是一个类别的协方差矩阵，例如 SIGMA(:,:,k)表示第k类的协方差矩阵
%     PI    Kx1列向量，第k个元素是类别k的先验概率
% 输出
%     P     Nx1列向量，第n个元素是高斯混合模型在点X(n,:)上的概率密度值

K = length( MU );

[ N , ~ ] = size( X );

P = zeros( N , K );

for n = 1 : N
    
    for k = 1 : K
        
        P( n , k ) = PI( k ) * mvnpdf( X( n , : ) , MU( k , : ) , SIGMA( : , : , k ) );
                            %利用第k类的均值向量、协方差矩阵
                            %调用mvnpdf计算多元高斯分布概率密度函数 N( X_n | MU_k , SIGMA_k ) 
                            %然后乘上第k类的先验概率PI_k，得到X_n属于第k类的概率类
    end
        
end

P = sum( P , 2 );           %对于X中的每一个点，将其属于每类的概率加起来，得到该点的概率密度

end
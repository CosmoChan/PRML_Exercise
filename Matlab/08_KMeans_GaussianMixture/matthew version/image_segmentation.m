function image = image_segmentation( image , K , alpha , max_iterations )
% image_segmentation.m
%     基于K-means聚类算法的图像分割程序 wuweizhen version
% 输入
%     image (height,width,d)三维数组 图像高度为height，宽度为width，颜色通道数为d
%     K     图像聚类的分割数目
%     alpha 位置信息的权重系数，alpha越大，位置信息对分割结果影响越大，当alpha=0时分割不考虑位置信息
%     max_iterations  最大迭代次数
% 输出
%     image (height,width,d)三维数组，经过分割后的图像

[ height , width , d ] = size( image );                    %获取图像高度、宽度、颜色通道数

[ x , y ] = meshgrid( 1 : height , 1 : width );            %设置一个网格，给图像每个像素设置一个坐标(x,y)

X = double( image );                                       %将图像由uint8格式转换为double格式

X( : , : , d+1 ) = x;                                      %在图像三维数组的第d+1页加上x坐标
    
X( : , : , d+2 ) = y;                                      %在图像三维数组的第d+2页加上y坐标

X = reshape( X ,  height * width , d+2 );                  %将图像由三维数组变形为矩阵，每行是一个观测值，每列是一个维

Std = std( X , [] , 1 ) + eps;                             %计算每个维的标准差

Std( d+1 : d+2 ) = Std( d+1 : d+2 ) / (alpha+eps);         %给位置的维数乘以权重系数alpha

X = bsxfun( @rdivide , X , Std );                          %归一化

[ labels , means , ~ ] = K_means( X , K , max_iterations );%进行K-means聚类

for k = 1 : K
                                                           %将每一类各个点的颜色，用该类的平均颜色来赋值
    X( labels==k , : ) = repmat( means( k , : ) , sum( labels == k ) , 1 );
    
end

X = bsxfun( @times , X , Std );                             %对数据进行预处理的逆变换，变换到原空间

X = reshape( X( : , 1 : d ) , height , width , d );         %只取出颜色信息，重新变形为(height,width,d)的三维数组形式

image = uint8( X );                                         %将类型转换为uint8

end
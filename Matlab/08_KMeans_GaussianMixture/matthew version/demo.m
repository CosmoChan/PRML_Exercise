% demo.m

load old_faithful.mat           %加载的数据中含有样本矩阵X，每行是一个观测值，每列是一维

Std = std( X , [] , 1 )+eps;    %求每个维的标准差

X_normalized = bsxfun( @rdivide , X , Std );%将样本的每个维除以相应地标准差，从而消去量纲，方差归一

K = 2;

max_iterations = 5000;

                                    %用归一化的样本进行聚类
[ labels , means , cost ] = K_means( X_normalized , K , max_iterations );

                                    %将均值进行逆归一化，变换到原空间
Means = bsxfun( @times , means , Std );

                                    %在原空间绘制聚类结果
plot( X( labels == 1 , 1 ) , X( labels == 1 , 2 ) , '.r' )

hold on

plot( Means( 1 , 1 ) , Means( 1 , 2 ) , '+r' )

plot( X( labels == 2 , 1 ) , X( labels == 2 , 2 ) , '.b' )

plot( Means( 2 , 1 ) , Means( 2 , 2 ) , '+b' )

hold off
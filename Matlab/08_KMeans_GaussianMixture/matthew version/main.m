% main.m
% 直接运行即可
% imshow函数需要高版本的matlab

load old_faithful.mat           %加载的数据中含有样本矩阵X，每行是一个观测值，每列是一维

Std = std( X , [] , 1 )+eps;    %求每个维的标准差

X_normalized = bsxfun( @rdivide , X , Std );%将样本的每个维除以相应地标准差，从而消去量纲，方差归一

%% K-Means
K = 2;

max_iterations = 5000;

subplot( 2 , 2 , 1 )

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

%% Gaussian Mixture

subplot( 2 , 2 , 2 )

max_iteration = 200;

K = 2;

threshold = 1e-10;

[ GAMMA , MU , SIGMA , PI , LogLikehood ] = Gaussian_Mixture_EM( X_normalized , K , threshold , max_iterations );

[ ~ , labels ] = max( GAMMA , [] , 2 );

y = GM_pdf( X_normalized , MU , SIGMA , PI );               %计算高斯混合模型在各个数据点上的概率密度值
 
plot3( X_normalized( labels == 1 , 1 ) , X_normalized( labels == 1 , 2 ) ,   y( labels == 1 ) , '.r' )

hold on

plot3( X_normalized( labels == 2 , 1 ) , X_normalized( labels == 2 , 2 ) ,   y( labels == 2 ) , '.b' )

MESH = 100;

[ mesh_x1 , mesh_x2 ] = meshgrid( linspace( min( X_normalized( : , 1 ) ) , max( X_normalized( : , 1 ) ) , MESH ) , linspace( min( X_normalized( : , 2 ) ) , max( X_normalized( : , 2 ) ) , MESH ) );

p = GM_pdf( [ reshape( mesh_x1 , MESH^2 , 1 ) , reshape( mesh_x2 ,MESH^2 , 1 ) ] , MU , SIGMA , PI );

meshc( mesh_x1 , mesh_x2 , reshape( p , MESH , MESH ) );
    
colormap gray                                               %取图像颜色为灰色

hold off

%% Image Segmentation

load IMG.mat

img = IMG{2};

subplot( 2 , 2 , 3 )

imshow( img )

K = 8;

alpha = 1;

max_iteration = 3000;

img = image_segmentation( img , K , alpha , max_iteration );

subplot( 2 , 2 , 4 )

imshow( img )
















% main.m
% ֱ�����м���
% imshow������Ҫ�߰汾��matlab

load old_faithful.mat           %���ص������к�����������X��ÿ����һ���۲�ֵ��ÿ����һά

Std = std( X , [] , 1 )+eps;    %��ÿ��ά�ı�׼��

X_normalized = bsxfun( @rdivide , X , Std );%��������ÿ��ά������Ӧ�ر�׼��Ӷ���ȥ���٣������һ

%% K-Means
K = 2;

max_iterations = 5000;

subplot( 2 , 2 , 1 )

                                    %�ù�һ�����������о���
[ labels , means , cost ] = K_means( X_normalized , K , max_iterations );

                                    %����ֵ�������һ�����任��ԭ�ռ�
Means = bsxfun( @times , means , Std );

                                    %��ԭ�ռ���ƾ�����
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

y = GM_pdf( X_normalized , MU , SIGMA , PI );               %�����˹���ģ���ڸ������ݵ��ϵĸ����ܶ�ֵ
 
plot3( X_normalized( labels == 1 , 1 ) , X_normalized( labels == 1 , 2 ) ,   y( labels == 1 ) , '.r' )

hold on

plot3( X_normalized( labels == 2 , 1 ) , X_normalized( labels == 2 , 2 ) ,   y( labels == 2 ) , '.b' )

MESH = 100;

[ mesh_x1 , mesh_x2 ] = meshgrid( linspace( min( X_normalized( : , 1 ) ) , max( X_normalized( : , 1 ) ) , MESH ) , linspace( min( X_normalized( : , 2 ) ) , max( X_normalized( : , 2 ) ) , MESH ) );

p = GM_pdf( [ reshape( mesh_x1 , MESH^2 , 1 ) , reshape( mesh_x2 ,MESH^2 , 1 ) ] , MU , SIGMA , PI );

meshc( mesh_x1 , mesh_x2 , reshape( p , MESH , MESH ) );
    
colormap gray                                               %ȡͼ����ɫΪ��ɫ

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
















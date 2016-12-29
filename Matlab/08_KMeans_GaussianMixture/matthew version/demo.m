% demo.m

load old_faithful.mat           %���ص������к�����������X��ÿ����һ���۲�ֵ��ÿ����һά

Std = std( X , [] , 1 )+eps;    %��ÿ��ά�ı�׼��

X_normalized = bsxfun( @rdivide , X , Std );%��������ÿ��ά������Ӧ�ر�׼��Ӷ���ȥ���٣������һ

K = 2;

max_iterations = 5000;

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
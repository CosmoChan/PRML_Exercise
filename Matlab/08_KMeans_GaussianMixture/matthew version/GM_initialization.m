function [ MU , SIGMA , PI ] = GM_initialization( X , K , max_iterations )
% GM_initialization.m
%     ��ϸ�˹ģ��EM�㷨�ĳ�ʼ������������K-means�㷨�����ɳ�ʼ������
%     �ú�������Gaussian_Mixture_EM�����Զ�����
% ����
%     X ��������������ÿ����һ���۲�����
%     K �Ƿ����Ŵ���
%     max_iterations K-means���е�����������
% ���
%     MU    ��ʼ���ľ�ֵ���󣬹���K�У�ÿ����һ����˹�ֲ��ľ�ֵ����
%     SIGMA ��ʼ����Э������ά���飬��ÿһҳ��һ����˹�ֲ���Э�������
%     PI    ��ʼ���������������������k��Ԫ������������k�����������

[ ~ , d ] = size( X );

[ labels , MU , ~ ] = K_means( X , K , max_iterations );

SIGMA = zeros( d , d , K );

PI = zeros( K , 1 );

for k = 1 : K
    
    SIGMA( : , : , k ) = cov( X( labels == k , : ) );
    
    PI( k ) = mean( labels == k );
    
end

end
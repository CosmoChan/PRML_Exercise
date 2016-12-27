function [ GAMMA , MU , SIGMA , PI , LogLikehood ] = Gaussian_Mixture_EM( X , K , threshold , max_iterations )
% Gaussian_Mixture_EM.m
%     ��˹���ģ��EM�㷨ʵ�ֳ��򣬲���K-means�㷨����ʼ��������ֵ wuweizhen version
% ����
%     X ��������������ÿ����һ���۲�����
%     K �Ƿ����Ŵ���
%     threshold ����ֹͣ������ֵ����������Ȼ�����仯��С��thresholdʱֹͣ����
%     max_iterations ����������
% ���
%     GAMMA ÿ���۲��������ڸ������ĺ�����ʣ�ÿ����һ��������ÿ����һ�����
%     MU  ��ֵ���󣬹���K�У�ÿ����һ����˹�ֲ��ľ�ֵ����
%     SIGMA Э�������ά���飬��ÿһҳ��һ����˹�ֲ���Э�������
%     PI  �����������������k��Ԫ������������k�����������
%     LogLikehood ������Ȼ����

[ N , ~ ] = size( X );                                                          %��ȡ������СN

[ MU , SIGMA , PI ] = GM_initialization( X , K , threshold );                   %��K-means�㷨��ʼ������

P = zeros( N , K );                                                             %����һ������P(n,k)��ʾ��n���۲��������k�ĸ���

LogLikehood_old = -inf;                                                         %��ʼ��������Ȼ����

for iterations = 1 : max_iterations

    for n = 1 : N
        
        for k = 1 : K                                                           
            %-------------------------------------------------------
            %�����n���۲�ֵ�������k�ĸ���
            %���õ�k��ľ�ֵ����MU(k,:)��Э�������SIGMA(:,:,k)�������k����˹�ֲ���x_n���ĸ����ܶ�ֵ��Ȼ����ϵ�k����������
            %����ʹ��matlab�Դ��Ķ�Ԫ��˹�ֲ������ܶȺ���������N(x_n|MU_k,SIGMA_k)����help mvnpdf��Ҳ���Ը��ݹ�ʽ�Լ���д
            
            P( n , k ) =          
            
            %-------------------------------------------------------                      
        end
        
    end

    LogLikehood = sum( log( sum( P , 2 ) ) , 1 );                               %���������Ȼ����
    
    fprintf('K:%i, iter: %i/%i, log likehood: %f\n', K , iterations , max_iterations , LogLikehood )%����Ļ����ʾÿ�ε�������Ϣ
    
    if LogLikehood - LogLikehood_old < threshold                                %���������Ȼ�����仯��С��ֵ��ֹͣ����

        break;
        
    end
    
    LogLikehood_old = LogLikehood;                                              %���¾ɵĶ�����Ȼ���������ڱȽ�ǰ�����α仯��
    %-------------------------------------------------------
    %�������ξ���GAMMA������GAMMA(n,p)��ʾ�۲�ֵx_n���ڵ�k��ĺ������
    %����N_k
    %������������������PI
    %���¸���ľ�ֵ����MU��ÿ����һ�����ÿ����һ��ά��
    %���¸���ľ�ֵ����MU������ÿ����һ�����ÿ����һ��ά��
    %��5�����㶼���Գ���ʹ��һ�д�����ɣ���5�У�
    
    
    
    
    
    
    
    
    
    
    
    
    
    %-------------------------------------------------------
    
    for k = 1 : K                                                               %�����k���Э�������
        
        delta = bsxfun( @minus , X , MU( k , : ) );                             %���ȼ�������۲����k���ֵ�����Ĳ����
        
        SIGMA( : , : , k ) = bsxfun( @times , delta' , GAMMA( : , k )' ) * delta / N_k( k );
                          %�ȼ��� delta' * diag( GAMMA( : , k )' ) * delta / %N_k( k )        
    end

end

end
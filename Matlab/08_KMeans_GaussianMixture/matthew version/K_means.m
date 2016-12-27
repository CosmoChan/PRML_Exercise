function [ labels , means , cost ] = K_means( X , K , max_iterations )
% K_means.m
%     K-means�㷨ʵ�ֳ��� wuweizhen version
% ����
%     X ��������������ÿ����һ���۲�����
%     K �Ƿ����Ŵ���
%     max_iterations ����������
% ���
%     labels ��ǩ����������ÿ��Ԫ�ر�ʾX����Ӧ���������
%     means  ��ֵ���󣬹���K�У�ÿ����һ���Ŵصľ�ֵ���������Ŵص����ĵ�
%     cost   ��ʧ����

[ N , d ] = size( X );                                      %��ȡ������N��ά��d

labels = ceil( K * rand( N , 1 ) );                         %�������N���ı�ǩ

means = zeros( K , d );                                     %�����������ڴ��K���Ŵؾ�ֵ����

for k = 1 : K
    means( k , : ) = mean( X( labels == k , : ) , 1 );      %���������ֵ����
end

changed = 1;                                                %����һ����¼�����Ƿ����ı�ı���
iterations = 0;                                             %����������
while changed == 1 && iterations < max_iterations           %�������ﵽ���޻��߾��಻�ı�ʱ��ֹͣ����
    
    changed = 0;
    
    for n = 1 : N                                           %�����n����
        %----------------------------------------------------
        %����x_n��K�����ľ�ֵ������ŷ�����룬����С�ߣ���x_n����������
        %���x_n������䶯����ô���¸���ľ�ֵ���������ҽ�changed�޸�Ϊ1
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        %----------------------------------------------------
    end
end

cost = 0;                                                   %���ñ�����¼��ʧ����

for k = 1 : K                                               %��ÿ���Ŵ��ϵ���ʧ����ֵ������
                                                            %�����k���Ŵصĸ����㵽�Ŵ����ĵĲ����
    delta = bsxfun( @minus , X( labels == k , : ) , means( k , : ) );
    
    distance = sum( delta.^2 , 2 );                         %�����k���Ŵصĸ����㵽�Ŵ����ĵ�ŷ������
    
    cost = cost + sum( distance );                          %��������ͣ��õ���k�����ʧ������Ȼ���ۼӵ��ܵ���ʧ������
    
end

end
function [ sigmoid ] = Sigmoid( a )
%Sigmoid ������ʹ��a��ֵ��0-1֮��
%   ����˵��
%   a ��һ���������߱������߾���
%   sigmoid ����׼��֮���a
sigmoid = 1 ./ (1 + exp(-a));

end


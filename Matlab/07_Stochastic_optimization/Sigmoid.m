function [ sigm ] = Sigmoid( a )
%Sigmoid ��Ϊ���������ز�ļ����
%   a ���������ز�������Ȩ��
%   sigm ���������ز�ļ���ֵ
sigm = 1 ./ (1 + exp(-a));
end
function [ a ] = softmax( Z )
%softmax ��Ϊ�����������ļ����
%   Z �����������������Ȩ��
%   a �����������ļ���ֵ
a = exp(Z) ./ (sum(exp(Z)));
end
function [ a ] = softmax( Z )
%softmax ʹ��softmax��������й�һ������
%   Z ����������������ֵ������ƫ����,2*11769
%   a ���������һ��ļ���ֵ
a = exp(Z) ./ (sum(exp(Z)));
end
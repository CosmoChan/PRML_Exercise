function cost = cross_entropy( Y , T )
% cross_entropy.m
%     ��softmax������㼤���ʱ���ý�������Ϊ���ۺ���
% ����
%     Y ��������softmax����ľ���ÿ����һ������
%     T ��ʵ�ķ����ǩT��ÿ����һ������
% ����
%     cost ΪY,T�����Ľ�����

[ n , ~ ] = size( Y ); 

cost = - sum( sum( T .* log( Y ) ) ) / n;

end
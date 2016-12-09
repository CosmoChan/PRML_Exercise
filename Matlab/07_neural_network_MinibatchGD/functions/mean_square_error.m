function cost = mean_square_error( Y , T )
% mean_square_error.m
%     ���ڻع����⣬ʹ�þ�������Ϊ���ۺ���
% ����
%     Y ��������������ľ���ÿ����һ������
%     T ��ʵ�ķ����ǩT��ÿ����һ������
% ����
%     cost ��� 1/2�������

[ N , ~ ] = size( Y );

cost = sum( sum( ( ( Y - T ) .^ 2 ) ) ) / ( 2 * N );

end
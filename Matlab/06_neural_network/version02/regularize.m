function T = regularize( Y )
%��Y��ÿ�е�����Ԫ�ر�Ϊ1�������Ϊ0

[ ~ , position ] = max( Y ,[] , 2 );    %��ȡYÿ���������Ԫ����

T = zeros( size( Y ) );                 %����0����

for i = 1 : size( Y , 1 )

    T( i , position( i ) ) = 1;         %��Yÿ�е����ֵ��Ϊ1

end

end
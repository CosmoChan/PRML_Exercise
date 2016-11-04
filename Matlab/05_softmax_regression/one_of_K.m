function labels = one_of_K( elements , targets )
% one_of_K��
%     ����targets��������elemets��ÿ��Ԫ�ر����"one-of-K"������ʽ�ı�ǩ
% ���룺
%     elements ��nά��������Ԫ�ؿ�������ֵ���������ַ���n����Ϊ1
%     targets ��Kά�������������з���Ŀ��
% ���
%     vector_labels K��n�����������i����elements�е�i��Ԫ�صı���
% ʾ����
%     ���룺
%               >>one_of_K( [ 2 1 2 3 2 1 0 2 3 ] , [ 0 1 2 3 ])
%     ��������
%               0  0  0  0  0  0  1  0  0
%               0  1  0  0  0  1  0  0  0
%               1  0  1  0  1  0  0  1  0
%               0  0  0  1  0  0  0  0  1
%      
%     ���룺
%               >>one_of_K( [ 'b' , 'd' ] , [ 'a' , 'b' , 'c' ] )
%     ��������
%               0  0
%               1  0
%               0  0

%��ȡ��ǩ����
n = length( elements );

%��ȡĿ���ǩ����������
m = length( targets );

%����յı�ǩ�������ڴ�ű�ǩ����
labels = zeros( m , n );

%��elements�е�ÿһ��Ԫ��������targetsƥ�䣬��������Ϊ����������
for i = 1 : n
    j = find( targets == elements(i) );
    labels( j , i ) = 1;
end

end

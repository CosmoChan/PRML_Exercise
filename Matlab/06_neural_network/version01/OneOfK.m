function [ T ] = OneOfK( Y )
%   OneOfK ����ǩ����OneOfK����
%   Y ������ı�ǩ���� 60000*1
%   T �Ѿ�����õı�ǩ���� 10*60000


% ����T��T��һ��10*60000�ľ��� 
T = zeros(10,size(Y,1));
for i=1:length(Y)
    if Y(i) == 0
        T(10,i) = 1;
    else
        T(Y(i),i) = 1;
    end
end


function [ T ] = OneOfK( Y )
%   OneOfK ����ǩ����OneOfK����
%   Y ������ı�ǩ����
%   T �Ѿ�����õı�ǩ����


% ������ǩ����T
T = zeros(10,size(Y,1));
% ���Y(i) = 0����T�ĵ�i�еĵ�10��λ�ø�ֵΪ1������T�ĵ�i�еĵ�Y(i)��λ�ø�ֵΪ1������Ϊ0
for i=1:length(Y)
    if Y(i) == 0
        T(10,i) = 1;
    else
        T(Y(i),i) = 1;
    end
end


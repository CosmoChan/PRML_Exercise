function [ T_train ] = SixToOne( T )
%SixToOne SixToOne ��T�е���6�ı��1���������0
%   ����˵��
%    T ����ı�ǩ����
T_train = zeros(size(T));
T_train(T == 6) = 1;
end


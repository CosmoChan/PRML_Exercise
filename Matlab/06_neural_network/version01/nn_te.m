function [ T_pred ] = nn_te( X, arg )
%nn_te ��������Գ���
%   X ��������
%   ������Ĳ���
%   T_pred Ԥ���ǩ����

% �������ļ����
a_1 = X;% 784*10000

z_2 = arg.W_1 * a_1 + arg.b_1;%100*10000
a_2 = Sigmoid(z_2);%100*10000

z_3 = arg.W_2 * a_2 + arg.b_2;%10*10000
a_3 = softmax(z_3);%10*10000

[n,m] = size(a_3);

T_pred = zeros(n,m);

[~,idx] = max(a_3);

for i=1:m
    T_pred(idx(i),i) = 1;
end

end
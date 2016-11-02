function [ T_pred ] = logistic_te( W,X_test )
%logistic_te �����������logistic����ó��Ĳ���W�Ͳ���������������ݵı�ǩԤ��ֵT_pred
%   ����˵����
%   W ѵ��ģ�͵õ��Ĳ���������һ��785*1������
%   X_test ѵ�����ݼ�������һ��1932*785�ľ���
%   T_pred ѵ�����ݵ�Ԥ���ǩֵ������һ��1932*1������
T = Sigmoid(X_test * W);
T_pred = zeros(size(T));
T_pred( T >= 0.5 ) = 1;
end


function [ W ] = logistic_tr( X_train,T_train,iteration,tolerance )
%logistic_tr �������������ţ�ٷ����ģ�͵Ĳ���W
%   ����˵��
%   X_train ������д����6��8�����ݵ�ѵ����������һ��11769*785�ľ���
%   T_train ѵ�����ı�ǩ������ѵ������Ϊ6�ı�ǩΪ1��ѵ������Ϊ8�ı�ǩΪ0������һ��1*11769������
%   iteration �����Ĵ���
%   tolerance ģ�Ͳ����������б�����
%   W ѵ���õ���ģ�Ͳ������⽫��һ��785*1������

% ����X_train�Ĵ�С
[~,n] = size(X_train);

% �����ʼ��W
W =0.01 * randn(n,1);

eps = 1e-10;

% cost_old = 0;
for tao=1:iteration
    
    %�������������ʵĹ���ֵ������һ��11769*1������
    y = Sigmoid( X_train * W );
    
    %������ʧ����
    cost = -sum( T_train' .* log( y - eps ) + ( 1 - T_train )' .* log( 1 - y + eps ));
%     if abs(cost - cost_old)<tolerance
%         break
%     end
%     
%     cost_old = cost;
    
    %����ԽǾ���R
    V = y .* (1 - y);
    R = diag( V );
    
    %����Ȩ��
    W = W - pinv(X_train' * R * X_train ) * X_train' * ( y - T_train');
    tao
    cost
end
end


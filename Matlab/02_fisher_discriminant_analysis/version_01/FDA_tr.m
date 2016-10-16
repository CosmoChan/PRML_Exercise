function [ W,w ] = FDA_tr( X_train,T_train )
% ����FDA_tr�Ǹ���ѵ�����ݺͱ�ǩ����ͶӰ����W���б���ֵw
% X_train �� T_train ��6��8��ѵ�����ݺͱ�ǩ

%�ֳ���ǩΪ6��8��ѵ����
X_train_6 = X_train(T_train ==6,:);
X_train_8 = X_train(T_train ==8,:);

%����ÿ���ÿ�����ݵĸ��������ľ�ֵ����������һ��1*785��������
m_six = mean(X_train_6);
m_eight = mean(X_train_8);

%����ÿ���Э�����������һ��785*785�ķ���
S_6 = (X_train_6-m_six)'*(X_train_6-m_six);
S_8 = (X_train_8-m_eight)'*(X_train_8-m_eight);

%����������ɢ�Ⱦ�������һ��785*785�ķ���
S_w = S_6 + S_8;

%����W������һ��785*1��������
W = pinv(S_w)*(m_six-m_eight)';

%����w
%pred_6 =X_train_6*W;
%pred_8 =X_train_8*W;
%m_6_pred = mean(pred_6);
%m_8_pred = mean(pred_8);
%w = -1/2*(m_6_pred+m_8_pred);
m_6_pred =m_six*W;
m_8_pred =m_eight*W;
w = -1/2*(m_6_pred+m_8_pred);
end


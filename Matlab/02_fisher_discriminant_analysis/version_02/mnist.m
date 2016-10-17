clear;clc;
[ train, test ] = load_mnist( true, 6, 8 );

train_X_6 = one_of_k( train.X(:,train.y == 6), true ); % ��ǩΪ6��ͼ�����ݲ�ƫ�� 785*5918����
train_X_8 = one_of_k( train.X(:,train.y == 8), true ); % ��ǩΪ8��ͼ�����ݲ�ƫ�� 785*5851����

[W, mean_6, mean_8 ] = FDA_tr( train_X_6, train_X_8 );

%����ֵ��
w0 = -(1/2)*( mean_6' * W + mean_8' * W );

y = FDA_te( test.X, W, w0 );

% ��׼ȷ��
accuracy = 1 - sum( abs( test.y - y )/2 )/(2 * length( test.y ) )

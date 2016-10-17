clear;clc;
[ train, test ] = load_mnist( true, 6, 8 );

train_X_6 = one_of_k( train.X(:,train.y == 6), true ); % 标签为6的图像数据并偏秩 785*5918矩阵
train_X_8 = one_of_k( train.X(:,train.y == 8), true ); % 标签为8的图像数据并偏秩 785*5851矩阵

[W, mean_6, mean_8 ] = FDA_tr( train_X_6, train_X_8 );

%求阈值：
w0 = -(1/2)*( mean_6' * W + mean_8' * W );

y = FDA_te( test.X, W, w0 );

% 求准确率
accuracy = 1 - sum( abs( test.y - y )/2 )/(2 * length( test.y ) )

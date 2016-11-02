close all;
clear;
clc;
tic
%% ˵��
% -mnist ����Logistic Regression(LR)����6��8��д����ʶ�������Ԥ������������Լ����ģ���ڲ��Լ��ϵ�׼ȷ�ʵĽű�

% ����˵����
  %û�в���
  
%% ѵ��ģ��
% ����ѵ�����ݼ�X_train������һ��784*60000�ľ���ÿһ����һ��ѵ������
X_train = loadMNISTImages('train-images-idx3-ubyte');

% ����ѵ�����ݼ��ı�ǩ����T_train������һ��60000*1������
T_train = loadMNISTLabels('train-labels-idx1-ubyte');

%�ҳ�y=6��y=8��ѵ������X_train(784*11769)��ѵ�����ݵı�ǩT_train(1*11769)
X_train = X_train(:,T_train == 6 | T_train == 8);
T_train = T_train(T_train == 6 | T_train == 8)';

% ��ѵ�����ݼ���ƫ�ñ���1������һ��11769*785�ľ���ÿһ����һ��ѵ������
X_train = [ones(1,size(X_train,2));X_train]';

% ��T_train��׼����ʹ�����6�ı��1���������0������һ��11769*1������
T_train = SixToOne(T_train);

% �������������ã��������200�Σ�ģ�͵�׼ȷ��Ϊ99.09%�������������õ���������
iteration = 200;

% ���õ���ֹͣ��tolerance
tolerance = 0.000001;

% ѵ��ģ�͵ó�ģ�͵Ĳ���W������һ��785*1�ľ���
W = logistic_tr(X_train,T_train,iteration,tolerance);

% �ͷ��ڴ�
clear X_train T_train;

%% ����ģ��
% �����������X_test������һ��784*10000�ľ���ÿһ����һ��ѵ������
X_test = loadMNISTImages('t10k-images-idx3-ubyte');

% ����������ݵı�ǩ����T_test������һ��10000*1������
T_test = loadMNISTLabels('t10k-labels-idx1-ubyte');

%�ҳ�y=6��y=8��ѵ������X_test(784*1932)��ѵ�����ݵı�ǩT_test(1*1932)
X_test = X_test(:,T_test == 6 | T_test == 8);
T_test = T_test(T_test == 6 | T_test == 8)';

% ���������ݼ���ƫ�ñ���1������һ��1932*785�ľ���ÿһ����һ��ѵ������
X_test = [ones(1,size(X_test,2));X_test]';

% ��T_test��׼��������һ��1932*1�ľ���
T_test = SixToOne(T_test);

% ����logistic_te.m�õ��������ݵ�Ԥ���ǩֵT_pred������һ��1932*1�ľ���
T_pred = logistic_te(W,X_test);

%�ͷ��ڴ�
clear X_test;

%% ����ģ���ڲ��������ϵ�׼ȷ��Ϊ99.39%
n = size(T_test,2);
accuracy =1- sum(abs(T_pred - T_test'))/(2 * n);

%�ͷ��ڴ�
clear T_test,T_pred;
toc
%close all;
%clear;
%clc;

addpath ../../../datasets

%% ���ݵ���ͳ�������
%����ѵ�����ݣ�X_train��һ��784*60000�ľ���T_train��һ��1*60000������
X_train=loadMNISTImages('train-images-idx3-ubyte');
T_train=loadMNISTLabels('train-labels-idx1-ubyte')';

%ѡ���ǩΪ6��8�����ݣ�X_train��һ��784*11769�ľ���T_train��һ��1*11769������
X_train = [ X_train(:,T_train==6), X_train(:,T_train==8) ];
T_train = [ T_train(T_train==6), T_train(T_train==8) ];
    
%��ѡ���ѵ�����ݼ���ƫ�ñ���1��X_train��һ��11769*784�ľ���
X_train = X_train';

%����������ݣ�X_test��һ��784*10000�ľ���T_test��һ��10000*1������
X_test=loadMNISTImages('t10k-images-idx3-ubyte');
T_test=loadMNISTLabels('t10k-labels-idx1-ubyte');

%ѡ���ǩΪ6��8�����ݣ�X_test��һ��784*1932�ľ���T_test��һ��1*1932������
X_test = [ X_test(:,T_test==6), X_test(:,T_test==8) ];
T_test = [ T_test(T_test==6)', T_test(T_test==8)' ];

%��ѡ��Ĳ������ݼ���ƫ�ñ���1��X_test��һ��1932*784�ľ���
X_test = X_test';

%�Բ������ݽ���������
Index = randperm(length(T_test));
T_test=T_test(Index);
X_test=X_test(Index,:);

%����������ݵ�����
n = size(T_test,2);
%% ģ��ѵ��
%ѵ��FDA_trģ�͵õ�ģ�Ͳ���W��w
[W,w] = FDA_tr(X_train,T_train);

%% ģ�ͼ���׼ȷ��
%ʹ��ѵ�����ݵõ�ģ�Ͳ���W��w����������ݵ�Ԥ���ǩ���õ���T_pred��һ��1932*1��������
T_pred = FDA_te(X_test,W,w);

%�������������ģ���е�׼ȷ�ʡ�������ΪԤ��ı�ǩֵ����6��8�����Ԥ����ȷ6-6=0,8-8=0�����Ԥ�����abs��6-8��=2����˳���2*n
%----------------------your code here------------------------





%------------------------------------------------------------

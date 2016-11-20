% mnist ���������������д���ֽ��з���
clear;
close all;
clc
tic
% ����ѵ������
train.X = loadMNISTImages('train-images-idx3-ubyte');
train.Y = loadMNISTLabels('train-labels-idx1-ubyte');

% ��Y����OneOfK���롣train.X��784*60000�ľ���train.Y��60000*1������
train.Y = OneOfK(train.Y);

% ����������ѧϰ����alpha
alpha = 2;

% �����������������iteration
iteration = 1500;

% ѵ��������ģ��
arg = nn_tr( train, alpha, iteration );

% �����������
test.X = loadMNISTImages('t10k-images-idx3-ubyte');
test.Y = loadMNISTLabels('t10k-labels-idx1-ubyte');

% ��Y����OneOfK���롣test.X��һ��784*10000����,test.Y��һ��10*10000�ľ���
test.Y = OneOfK(test.Y);

% Ԥ���������,10*10000
Y_pred = nn_te(test.X, arg);

% ����׼ȷ��accuracy��׼ȷ����95.7%���ң�ʱ��551s��
%��������ʱ�䣬�����ʵ����ѧϰ����alpha�����Ҽ��ٵ�������iteration
accuracy=1 - sum(sum(abs(test.Y - Y_pred)))/(2 * size(test.Y,2));
toc
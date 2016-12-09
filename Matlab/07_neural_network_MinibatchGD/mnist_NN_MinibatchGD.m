function mnist_NN_MinibatchGD
% �����������ݶ��½�ʵ����ȳ��� wuweizhen version
%
% ������ʹ�ð��޷Ż�˳��ѭ����С�����ݶ��½�����֮ǰ�������ݶ��½���ȣ�ֻ��Ҫ����С���޸�
% ˳��ѡȡС������ϸ�����NN_train�����鳢��ʹ�ò�ͬ������Ż�����
% ������Ĭ��ʹ�õ�������������ֹͣ��Ҳ���鳢�Բ��������ĵ���ֹͣ����
%
% ������
%    digits ��һ�����ֵ��������������з����Ŀ������
%    config ��һ��������������ά������������Ĳ���L�����Ԫ�ش�С���Ƹ���ĵ���Ԫ����
%    activations Ԫ�����飬���� L-1 ���������������˶�Ӧ��2,3,...,L��ļ����
%    derivatives Ԫ�����飬���� L-1 ���������ľ�������˶�Ӧ��2,3,...,L��ļ����
%    cost_function �������������ʹ�õ���ʧ����
%    max_epochs ������������ѵ���������ѵ�������������������������������Ƶ���ֹͣ
%    eta �Ǹ�����Ϊѧϰ���ʡ�������̫������ʧ�����������𵴣�������������̫С����ѧϰЧ�ʵ�
% �����
%    Wb ѵ�����õ��Ĳ������� W , b
%    accuracy ������ȷ��
% ʾ��
%    ��0-9���з��࣬�������784����Ԫ��һ����50����Ԫ�����ز㣬�������10����Ԫ��
%    ���ز㼤���Ϊtanh������㼤���Ϊsoftmax��ѧϰ����Ϊ0.03������СΪ10����������������Ϊ5��
%    ������1���������ѵ�������Ҳ�����ȷ��ԼΪ96%
%    ������ʾ������

% ��������
addpath functions

digits = [ 0 1 2 3 4 5 6 7 8 9 ];

struct = [ 784 50 10 ];

activations = { @tanh , @softmax };            %�����

derivatives = { @diff_tanh , @(x)1 };          %������ĵ�����

cost_function = @cross_entropy;                %ʹ�ý�������ʧ����

max_epochs = 5;                                %����ѵ�������ĵ�����������

minibatch_size = 10;                           %С�����ݶ��½���ÿ���Ĵ�С

learning_rate = 0.03;                          %ѧϰ����

%����ѵ�����ݺͲ�������
[ train , test ] = load_mnist( digits );

%���������Xת��Ϊÿ��һ��������ÿ��Ϊһ��ά��
train.X = train.X';
test.X = test.X';

%����������ı�ǩy����"one-of-K"���룬����������digits����
train.y = one_of_K( train.y , digits )';
test.y = one_of_K( test.y , digits )';

tic;
%����ѵ�����õ���������Wb�����������W��b
Wb = NN_train( train.X , train.y , struct , activations , derivatives , cost_function , minibatch_size , max_epochs , learning_rate );
train_time = toc;

%��ϵ������Wb�����������test.X����������ֵ
hat_Y = NN_test( test.X , Wb , activations );
    
%����������ֵ�������滯Ϊone-of-K����ı�ǩ����
hat_T = regularize( hat_Y );
    
%���������ȷ�ʣ�������Ԫ��accuracy�ĵ� i ��λ��
n = size( test.y , 1 );
accuracy = 1 - sum( sum( abs( test.y - hat_T ) ) ) / ( n * 2 );

fprintf('train time: %.2fs, accuracy: %.2f%%\n' , train_time , accuracy*100 );

%���ؽ��
%result = [ Wb , accuracy ];

end
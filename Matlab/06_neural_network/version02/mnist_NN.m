function result = mnist_NN
% ������ʵ����ȳ���
%
% ������
%    digits ��һ�����ֵ��������������з����Ŀ������
%    config ��һ��������������ά������������Ĳ���L�����Ԫ�ش�С���Ƹ���ĵ���Ԫ����
%    activations Ԫ�����飬���� L-1 ���������������˶�Ӧ��2,3,...,L��ļ����
%    derivatives Ԫ�����飬���� L-1 ���������ľ�������˶�Ӧ��2,3,...,L��ļ����
%    max_iterations ���������������������������������������������Ƶ���ֹͣ
%    eta �Ǹ�����Ϊѧϰ���ʡ�������̫������ʧ�����������𵴣�������������̫С����ѧϰЧ�ʵ�
% �����
%    result ��һ��Ԫ�����飬����
%         Wb ѵ�����õ��Ĳ������� W , b
%         accuracy ������ȷ��
% ʾ��
%    ��0-9���з��࣬�������784����Ԫ��һ����100����Ԫ�����ز㣬�������10����Ԫ��
%    ���ز㼤���Ϊtanh������㼤���Ϊsigmoid��ѧϰ����Ϊ1������200�εĵ�����
%    ���������ʾ������

% ��������
addpath Activations_functions

digits = [ 0 1 2 3 4 5 6 7 8 9 ];

config = [ 784 100 10 ];

activations = { @tanh , @sigmoid };

derivatives = { @diff_tanh , @diff_sigmoid };

max_iterations = 200;

eta = 2;

%����ѵ�����ݺͲ�������
[ train , test ] = load_mnist( digits );

%���������Xת��Ϊÿ��һ��������ÿ��Ϊһ��ά��
train.X = train.X';
test.X = test.X';

%����������ı�ǩy����"one-of-K"���룬����������digits����
train.y = one_of_K( train.y , digits )';
test.y = one_of_K( test.y , digits )';

%����ѵ�����õ���������Wb�����������W��b
Wb = NN_train( train.X , train.y , config , activations , derivatives , max_iterations , eta );
   
%��ϵ������Wb�����������test.X����������ֵ
hat_Y = NN_test( test.X , Wb , activations );
    
%����������ֵ�������滯Ϊone-of-K����ı�ǩ����
hat_T = regularize( hat_Y );
    
%���������ȷ�ʣ�������Ԫ��accuracy�ĵ� i ��λ��
n = size( test.y , 1 );
accuracy = 1 - sum( sum( abs( test.y - hat_T ) ) ) / ( n * 2 );

result = [ Wb , accuracy ];

end
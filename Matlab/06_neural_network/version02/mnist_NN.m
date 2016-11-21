function result = mnist_NN( digits , config , activations , derivatives , max_iterations , eta )
% ������ʵ����ȳ���
%
% ���룺
%    digits ��һ�����ֵ��������������з����Ŀ������
%    config ��һ��������������ά������������Ĳ���L�����Ԫ�ش�С���Ƹ���ĵ���Ԫ����
%    activations Ԫ�����飬���� L-1 ���������������˶�Ӧ��2,3,...,L��ļ����
%    dervatives Ԫ�����飬���� L-2 ���������ľ�������˶�Ӧ��2,3,...,L-1��ļ����
%    max_iterations ���������������������������������������������Ƶ���ֹͣ
%    eta �Ǹ�����Ϊѧϰ���ʡ�������̫������ʧ�����������𵴣�������������̫С����ѧϰЧ�ʵ�
% �����
%    result ��һ��Ԫ������
%        ��һ����
%            Wb_LIST ��max_iterations�У���i���ǵ� i �ε��������ĸ����ϵ�� W �� b
%        �ڶ�����
%            accuracy ��max_iterations�У���i���ǵ� i �ε���������ϵ�����ϵĲ�����ȷ��
%
% ʾ��
%    ��0-9���з��࣬�������784����Ԫ��һ����100����Ԫ�����ز㣬�������10����Ԫ��
%    ���ز�������ļ����Ϊsigmoid���������ز㵼����Ϊdiff_sigmoid����������50�εĵ�����
%    ѧϰ����Ϊ0.7��������õ�ʾ������
%
%    >> Activations = { @sigmoid , @sigmoid };
%    >> Dervatives = { @diff_sigmoid };
%    >> Digits = [ 0 1 2 3 4 5 6 7 8 9 ];
%    >> Config = [ 784 100 10 ];
%    >> Max_iterations = 1000;
%    >> Eta = 0.7;
%    >> Result = mnist_NN( Digits, Config, Activations, Dervatives, Max_iterations, Eta )

%����ѵ�����ݺͲ�������
[ train , test ] = load_mnist( digits );

%���������Xת��Ϊÿ��һ��������ÿ��Ϊһ��ά��
train.X = train.X';
test.X = test.X';

%����������ı�ǩy����"one-of-K"���룬����������digits����
train.y = one_of_K( train.y , digits )';
test.y = one_of_K( test.y , digits )';

%����ѵ�����ò��轫�������ɸ���������������ϵ���б�Wb_LIST
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
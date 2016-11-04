function result = softmax_mnist( digits , lambda )
% softmax_mnist:
%     ����softmax�����Ķ���logistic�ع��ʵ�ֳ�������Newton-Raphson��������
%     ����ܽ�����⡣
% ���룺
%     digits ��һ��Kά����������K������ͬ��Ŀ������
%     lambda ��һ���Ǹ�����Ϊ����ϵ��
% �����
%     result ��һ��Ԫ����������
%          W Ϊd��K�еľ���ģ�Ͳ�������ÿһ�ж���digits�ж�Ӧ����Ĳ���
%          norm(W) �ǲ�������W�Ķ�����
%          iterations ��ѵ�����ĵ�������
%          accuracy ��ѵ��ģ���ڸ������������µ���ȷ��
%
% ʵ����
%      ���룺
%          >> softmax_mnist( [ 0 1 2 3 4 5 6 7 8 9 ] , 200 )
%      �����
%          ans = 
%
%              [785x10 double]    [2.47962147181502]    [6]    [0.9231]
%
%      �������0-9�����ֽ��з����ʵ���У�ѵ���õ���ģ�Ͳ���W��785x10�ľ���
%      W�Ķ�����Լ��2.47962147181502��������6�ε��������ڲ������ݵ�Ԥ����ȷ
%      ��Ϊ92.31%�����⣬�������к�ʱԼ40���ӡ�

%����ѵ�����ݺͲ�������
[ train , test ] = load_mnist( digits );

%��ѵ������������x���볣��Ԫ��1��Ϊ�ؾ�
%��ѵ�����ݵ��������Xת��Ϊÿ��һ��������ÿ��Ϊһ��ά��
train.X = [ ones( 1 , length(train.y) ) ; train.X ]';

%��ѵ�����ݵı�ǩy����"one-of-K"���룬����������digits����
train.y = one_of_K( train.y , digits )';

%ͨ��ѵ�����ݵ������������Ӧ�ı�ǩ�����ڸ���������ϵ����
%��ģ�Ͳ�������W���������������iterations
[ W , iterations ] = softmax_train( train.X , train.y , lambda );
%�ͷ��ڴ�
clear train;

%����������������x���볣��Ԫ��1��Ϊ�ؾ�
%���������ݵ��������Xת��Ϊÿ��һ��������ÿ��Ϊһ��ά��
test.X = [ ones( 1 , length(test.y) ) ; test.X ]';

%��ѵ�����ݵı�ǩy����"one-of-K"���룬����������digits����
test.y = one_of_K( test.y , digits )';

%����ģ�Ͳ�������W���Բ������ݼ��ϵľ����������Ӧ��ǩ���Ͼ���

hat_T = softmax_test( test.X , W );

%��ʵ��ǩ����test_y��Ԥ���ǩ����y_hat������������������������������õ��������
n = size( test.y , 1 );
accuracy = 1 - sum(sum(abs( test.y - hat_T ))) / ( n * 2 );

%��� ��������������W �� ����W�Ķ����� �� ѵ������������ �� ������ȷ��
result={ W , norm(W) , iterations , accuracy };

end


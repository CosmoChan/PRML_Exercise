function Wb = NN_train( X , T , struct , activations , derivatives , cost_function , minibatch_size , max_epochs , eta  )
% ������ѵ������
%
% ���룺
%    X ���������������ÿ����һ������
%    T ���������������ÿ����һ������
%    config ��һ��������������ά������������Ĳ���L�����Ԫ�ش�С���Ƹ���ĵ���Ԫ����
%    activations ��һ��Ԫ�����飬���� L-1 �� �������������˶�Ӧ��2,3,...,L��ļ����
%    dervatives ��һ��Ԫ�����飬���� L-1 ���뼤�����Ӧ�� �������ľ��
%    max_iterations ��һ�����������������������������������������������Ƶ���ֹͣ
%    eta ��һ���Ǹ�����Ϊѧϰ���ʡ�������̫������ʧ�����������𵴣�������������̫С����ѧϰЧ�ʵ�
% �����
%    Wb �������ϣ����������W���Ϻ�b����

L = length( struct );                 %ȡ������㵽�������ܲ���
[ N , ~ ] = size( X );                %ȡ������С
if minibatch_size > N
    error('minibatch���ɴ���ѵ����������')
end
max_iterations = ceil( max_epochs * N / minibatch_size );%�������������� ����ѵ���������� * �������� / С������

activations = [ {[]} , activations ]; %��������㲻���ü������ǰ����ӿ�Ԫ����ռλ
derivatives = [ {[]} , derivatives ]; %ͬ��

Z = cell( L , 1 );                    %��Ÿ���ĵ�Ԫ����������е�һ��������ռλ
A = cell( L , 1 );                    %��Ÿ���ĵ�Ԫ�����������
Delta = cell( L , 1 );                %��Ÿ���Ĳв�������е�һ��������ռλ
errors = zeros( max_iterations , 1 ); %��������errors�����ڼ�¼ÿ�εĵ�������ʧ����ֵ
                                      
W = cell( L - 1 , 1 );                %���ǰL-1���ϵ������ W
b = cell( L - 1 , 1 );                %���ǰL-1���ƫ��ϵ������ b
for l = 1 : L-1                       %��ʼ��������� W b
    r = sqrt( 6 / ( struct( l ) + struct( l + 1 ) ) );          % xavier���鹫ʽ
    W{ l } = 2 * r * rand( struct( l ) , struct( l + 1 ) ) - r;  
    b{ l } = zeros( 1 , struct( l + 1 ) );
end

batch_head = 1;                         %minibatch������������
iterations = 0;                         %С�����ĵ�������
go_on = 1;                              %go_on��Ԥ���ĵ���ָֹͣ��ģ����Գ�����ӵ���ֹͣ���� 
while iterations < max_iterations && go_on
    iterations = iterations + 1;
    
    batch_tail = batch_head + minibatch_size - 1;  %��������X��T�У���˳��ѭ����ȡminibatch_size��ѵ������
    if batch_tail <= N
        A{ 1 } = X( batch_head : batch_tail , : );
        batch_T = T( batch_head : batch_tail , : );
    else
        batch_tail = batch_tail - N;
        A{ 1 } = [ X( batch_head : end , : ) ; X( 1 : batch_tail , : ) ];
        batch_T = [ T( batch_head : end , : ) ; T( 1 : batch_tail , : ) ];
    end
    batch_head = mod( batch_tail , N ) + 1;

    for l = 1 : L-1                           %��1��L-1�㣬����ǰ�򴫲�
        
        Z{ l + 1 } = bsxfun( @plus , A{ l } * W{ l } , b{ l } );
        
        A{ l + 1 } = activations{ l + 1 }( Z{ l + 1 } );
        
    end                               %�������ֵΪ Y := A{ L }
    
    Delta{ L } = derivatives{ L }( Z{ L } ) .* ( A{ L } - batch_T );%�������һ��Ĳв����
    
    errors( iterations ) = cost_function( A{ L } , batch_T );  %������ʧ����
    
    fprintf('epochs: %.2f, cost: %f\n', iterations * minibatch_size / N , errors( iterations ) );
    
    for l = L-1 : -1 : 1                      %�ӵ�L-1����1�㣬�������򴫲�
        
        Gradient_W = A{ l }' * Delta{ l + 1 } / minibatch_size;%�����l��ϵ������ W ���ݶ�
        
        Gradient_b = sum( Delta{ l + 1 } ) / minibatch_size;   %�����l��ϵ������ b ���ݶ�
        
        W{ l } = W{ l } - eta * Gradient_W;            %��ϵ������ W �����ݶ��½�
        
        b{ l } = b{ l } - eta * Gradient_b;            %��ƫ��ϵ������ b �����ݶ��½�

        if l ~= 1                              %���l���ǵ�һ�㣬��ô����ò�Ĳв����
            
            Delta{ l } = derivatives{ l }( Z{ l } ) .* ( Delta{ l + 1 } * W{ l }' );
            
        end
        
    end
    
end

plot( errors , '.b' , 'MarkerSize',3 )         %������������iterations�仯�ĺ���ͼ��
xlabel('iterations')
ylabel('cost')

Wb = { W , b };

end
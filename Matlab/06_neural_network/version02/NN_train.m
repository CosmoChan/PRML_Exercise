function Wb = NN_train( X , T , config , activations , derivatives , max_iterations , eta  )
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

L = length( config );                 %ȡ������㵽�������ܲ���
[ N , ~ ] = size( X );                %ȡ������С
eta = eta / N;                        %���ݶ��½���������������СN

activations = [ {[]} , activations ]; %��������㲻���ü������ǰ����ӿ�Ԫ����ռλ
derivatives = [ {[]} , derivatives ]; %ͬ��

Z = cell( L , 1 );                    %��Ÿ���ĵ�Ԫ����������е�һ��������ռλ
A = cell( L , 1 );                    %��Ÿ���ĵ�Ԫ�����������
Delta = cell( L , 1 );                %��Ÿ���Ĳв�������е�һ��������ռλ
errors = zeros( max_iterations , 1 ); %��������errors�����ڼ�¼ÿ�εĵ�������ʧ����ֵ
                                      
W = cell( L - 1 , 1 );                %���ǰL-1���ϵ������ W
b = cell( L - 1 , 1 );                %���ǰL-1���ƫ��ϵ������ b
for l = 1 : L-1                       %��ʼ��������� W b
    W{ l } = 0.1 * randn( config( l ) , config( l + 1 ) );  
    b{ l } = 0.1 * randn( 1 , config( l + 1 ) );
end

A{ 1 } = X;                           %���õ�һ�㵥Ԫ��������� A Ϊ X

for iterations = 1 : max_iterations   %��ʼ���������ָ������֮������ѭ��

    iterations                                %�����������

    for l = 1 : L-1                           %��1��L-1�㣬����ǰ�򴫲�
        
        Z{ l + 1 } = bsxfun( @plus , A{ l } * W{ l } , b{ l } );
        
        A{ l + 1 } = activations{ l + 1 }( Z{ l + 1 } );
        
    end                               %�������ֵΪ Y := A{ L }
    
    Delta{ L } = derivatives{ L }( Z{ L } ) .* ( A{ L } - T );%�������һ��Ĳв����
    
    errors( iterations ) = sum(sum(( A{ L } - T ).^2 )) / ( 2 * N );  %�������ƽ������ʧ����
                                                        %���ڻع����⣬ʹ�����ƽ���ͣ��������⣬��ʹ�ý�����
    errors( iterations )                      %�����ʧ����
    
    for l = L-1 : -1 : 1                      %�ӵ�L-1����1�㣬�������򴫲�
        
        Gradient_W = A{ l }' * Delta{ l + 1 };         %�����l��ϵ������ W ���ݶ�
        
        Gradient_b = ones( 1 , N ) * Delta{ l + 1 };   %�����l��ϵ������ b ���ݶ�
        
        W{ l } = W{ l } - eta * Gradient_W;            %��ϵ������ W �����ݶ��½�
        
        b{ l } = b{ l } - eta * Gradient_b;            %��ƫ��ϵ������ b �����ݶ��½�

        if l ~= 1                              %���l���ǵ�һ�㣬��ô����ò�Ĳв����
            
            Delta{ l } = derivatives{ l }( Z{ l } ) .* ( Delta{ l + 1 } * W{ l }' );
            
        end
        
    end
    
end

plot( errors , '.b' )                  %������������iterations�仯�ĺ���ͼ��

Wb = { W , b };

end
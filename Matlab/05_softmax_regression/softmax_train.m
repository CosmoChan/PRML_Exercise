function [ W , iterations ] = softmax_train( X , T , lambda )
% softmax_train��
%     ����ѵ�����ݵ������������Ͼ���X�ͱ�ǩ���Ͼ���T���Լ�����ϵ��lambda
%     ����Newton-Raphson�������softmaxģ�͵Ĳ�������W���������������
% ���룺
%     X ��n��d�о���ÿ����һ������������ÿ������������dά
%     T ��n��K�о���ÿ���Ǳ�ǩ��ÿ�е�K��Ԫ������һ��Ϊ1������Ϊ0
%     lambda ������ϵ������ֹ����W�Ķ���������
% �����
%     W ��d��K�о���K����ÿ�ж�����Ӧ���Ĳ�������
%     iterations ��������

%��ȡ����������ά��
[ ~ , d ] = size( X );

%��ȡ�������
[ ~ , K ] = size( T );

%���ò���W�ĵ�����ֵ��Ϊ�ӽ�0��d*K��1�е�����
W = rand( d * K , 1 )*0.01;%%

%�������裬����������Ϊ15
for iterations = 1 : 15

    %��softmax��������ÿ����������K����ÿһ��ĸ��ʣ�ģ�Ͳ���W��d*K��1�е�����
    Y = softmax_hypothesis_function( X , W , 1 );

    %�����ʾ���ͷ����ǩ��������
    Delta = Y - T;

    %����K���У�������ݶ�������ƴ�ӳ�һ��d*k�е��ݶ�������
    Gradient = zeros( d * K , 1 );
    
    for j = 1 : K
        %�����j��Ĳ���w_j
        Gradient( 1 + (j-1)*d : j*d ) = sum( bsxfun( @times , X' , Delta( : , j )') , 2 );       
    end
    
    %�ݶ������м���������
    Gradient = Gradient + lambda * W;
    
    %����Hessian�����K��K���ӿ飬��ƴ�ӳ�d*K��d*K�е�Hessian����
    Hessian = zeros( K * d );
    for k = 1 : K   
        for j = 1 : K         
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %������������հ״�
            

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %����k,j���ӿ����Hessian����Ӧλ��
            Hessian( 1+(k-1)*d : k*d , 1+(j-1)*d : j*d ) =  Sub_hessian;
            
        end
    end
    
    %Hessian�����м���������
    Hessian = Hessian + lambda * eye( K * d );
    
    %����ģ�Ͳ���������W��
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %��������հ״�����ʾ������ʱ��A\B����inv(A)*B

    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %������ֹ�жϣ����ݶ�Gradient�Ķ������ӽ�0��ֹͣ����
    if norm( Gradient ) < 200
        break;        
    end
    
end

%��ģ�Ͳ���������Wת����d��K�еľ�����Ϊ������
W = reshape( W , d , K );

end


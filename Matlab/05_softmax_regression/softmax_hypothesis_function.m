function Y = softmax_hypothesis_function( X , W , is_column )
%  softmax_hypothesis_function�� 
%     softmax����������ÿ����������K����ÿһ��ĸ���
% ���룺
%     X ���������ݾ�����X��n��d�У����ʾ��n��������ÿ��������d��ά��
%     W ��softmaxģ�͵Ĳ���������K�������⣬W������������Ҫ��
%             ���������is_columnȱʡ����Ϊ0��ʱ��W��d��K�еľ���
%             ���������is_columnΪ����ֵ������ֵ��ʱ��W��d*K��1�е�����
% �����
%     Y Ϊn��K�о�����Ԫ��Y(i,j)��ʾX�е�i�е��������ڵ�j��ĸ���

%���W��d*K��1�в�����������ô����ת��Ϊd��K�еľ���
if nargin == 3 && is_column ==1
    
    %��ȡ����������ά��
    [ ~ , d ] = size( X );
    
    %��ȡ��������K
    K = length( W ) / d;
    
    %��d*K��1�е�����ת����d��K�еľ���
    W = reshape( W , d , K );
else   
    %��������W��d��K�о�����ôֱ�ӻ�ȡ��������K
    [ ~ , K ] = size( W ); 
end

Y = exp( X * W );

%Y��ÿһ����ÿһ��Ԫ�ض����Ը��еĺͣ��Ӷ����ʹ�һ��
Y = bsxfun( @rdivide , Y , sum( Y , 2 ) );

%S = sum( Y , 2 );
%for i = 1 : K    
%    Y(:,i) = Y(:,i) ./ S;    
%end

end


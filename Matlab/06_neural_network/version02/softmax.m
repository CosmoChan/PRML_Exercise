function Z = softmax( a )

Z = exp( a );

%Y��ÿһ����ÿһ��Ԫ�ض����Ը��еĺͣ��Ӷ����ʹ�һ��
Z = bsxfun( @rdivide , Z , sum( Z , 2 ) );

end


function dydx = diff_softmax( X )

%softmax���������ڸ�ÿ������
%Ҫ��ÿ����һ����������

Exp_X = exp( X );

S = sum( Exp_X , 2 );

dydx = Exp_X .* ( bsxfun( @minus , S , Exp_X ) );

dydx = bsxfun( @rdivide , dydx , S.^2 );

end
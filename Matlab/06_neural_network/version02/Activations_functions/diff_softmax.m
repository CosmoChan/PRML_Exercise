function dydx = diff_softmax( X )

%softmax函数，对于各每个分量
%要求每行是一个输入向量

Exp_X = exp( X );

S = sum( Exp_X , 2 );

dydx = Exp_X .* ( bsxfun( @minus , S , Exp_X ) );

dydx = bsxfun( @rdivide , dydx , S.^2 );

end
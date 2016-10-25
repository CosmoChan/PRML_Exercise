function [ y ] = perceptron_te( w, test_X, C1, C2 )

a = w * addbias( test_X, true );
y = activate_function( a );
y( y == 1 ) = C1;
y( y == -1 ) = C2;

end
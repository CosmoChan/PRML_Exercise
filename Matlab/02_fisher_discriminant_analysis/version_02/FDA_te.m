function [ y ] = FDA_te( test_X, W, w0 )

y0 = W' * one_of_k( test_X, true );
y = y0;
y( y > w0 ) = 6;
y( y < w0 ) = 8;

end
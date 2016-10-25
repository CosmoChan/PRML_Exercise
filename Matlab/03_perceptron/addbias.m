function [ X ] = addbias( X0, binary_digits )
if binary_digits == false;
    [ m, n ] = size( X0 );
    vone = ones( m, 1 );
    X = [ vone, X0 ];
end
[ m, n ] = size( X0 );
vone = ones( 1, n );
X = [ vone; X0 ];
end
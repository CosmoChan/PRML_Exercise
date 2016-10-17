function [ X ] = addbias( X0, binary_digits )
% This function is to add bias 1 on the beginning of X0,
% Input :   X0 is the target we need to add bias.
%           binary_digits is a logical digits.
% if binary_digits == false, we add 1 on the beginning of each row of X0;
% if binary_digits == true, we add 1 on the beginning of each column of X0;
% the default condition is binary_digits == true.

if binary_digits == false;
    [ m, n ] = size( X0 );
    vone = ones( m, 1 );
    X = [ vone, X0 ];
end
[ m, n ] = size( X0 );
vone = ones( 1, n );
X = [ vone; X0 ];

end
function [ f ] = activate_function( a )
f = a;
for i = 1 : length( a );
    if a( i ) >= 0
        f( i ) = 1;
    else
        f( i ) = -1;
    end
end
end
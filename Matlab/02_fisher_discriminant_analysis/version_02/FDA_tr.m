function [ W, mean_6, mean_8 ] = FDA_tr( train_X_6, train_X_8 )

% 785*1¾ØÕó
mean_6 = mean( train_X_6' )';
mean_8 = mean( train_X_8' )';

[ m1, n1 ] = size( train_X_6 );
[ m2, n2 ] = size( train_X_8 );
st_6 = zeros( m1, n1 );
st_8 = zeros( m2, n2 );
for i = 1:n1
    st_6(:,i) = train_X_6( :, i ) - mean_6;
end
for i = 1:n2
    st_8(:,i) = train_X_8( :, i ) -  mean_8;
end

%785*785Ğ­·½²î¾ØÕó
S_6 = ( st_6 ) * ( st_6 )';
S_8 = ( st_8 ) * ( st_8 )';
% S_6 = ( train_X_6 - repmat( mean_6, m1, n1 ) )*( train_X_6 - repmat( mean_6, m1, n1 ) )';
% S_8 = ( train_X_8 - repmat( mean_8, m2, n2 ) )*( train_X_8 - repmat( mean_8, m2, n2 ) )';

S_w = S_6 + S_8;

W = pinv( S_w ) * ( mean_6 - mean_8 );

end

function [ W, mean_6, mean_8 ] = FDA_tr( train_X_6, train_X_8 )
% This is a function to train the model.
% Input:  train_X_6 is the image data of number 6;
%         train_X_8 is the image data of number 8;
% Output: W is the parameters of FDA
%         mean_6 is the mean of the image data of number 6;
%         mean_8 is the mean of the image data of number 8;

% To calculate mean_6 and mean_8
mean_6 = mean( train_X_6' )';
mean_8 = mean( train_X_8' )';

% To calculate 
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% To do: calculate the convariance matrix of number 6 ( S_6 ) and 8 ( S_8 ) individually 
% using st_6 and st_8.
%
% Your code goes here:
S_6 = ( st_6 ) * ( st_6 )';
S_8 = ( st_8 ) * ( st_8 )';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To calculate S_w
S_w = S_6 + S_8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% To do: please add code to calculate the parameters W using the formular
% we've learnt
%
% Your code goes here:
W = pinv( S_w ) * ( mean_6 - mean_8 );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

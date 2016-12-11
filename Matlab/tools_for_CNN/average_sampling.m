function samp = average_sampling( X , sampling_size )
%Æ½¾ù³Ø»¯

samp = convn( X , ones( sampling_size ) , 'valid' ) / sampling_size.^2;

samp = samp( 1 : sampling_size : end , 1 : sampling_size : end , : );

end
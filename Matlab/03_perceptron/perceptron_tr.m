function [ w ] = perceptron_tr( train_X, train_y, C1, C2 )
t = train_y;
t( train_y == C1 ) = 1;
t( train_y == C2 ) = -1;

X = addbias( train_X, true );

% To default the parameters
m = size( X );
w = ones( 1, m( 1 ) );
E = 1;
lambda = 1;
i = 1;

% To set the iteration termination and limit the iteration times
while E ~= 0 && i < 1000
    C = w * X .* t < 0;
    t_n = t( C == 1 );
    phi_n = X( :, C == 1 );
    E = - w * phi_n * t_n';
    w = w + lambda * ( phi_n * t_n' )';
    i = i + 1;
end

end
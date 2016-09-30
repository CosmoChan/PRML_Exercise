function T = oneOfK(y)
%
T = zeros(length(y), 2);

for i = 1:length(y)
    if round(y(i)) == 0
        T(i, 1) = 1;
    else
        T(i, 2) = 1;
    end
end


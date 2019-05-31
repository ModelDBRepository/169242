
%% Numerical simulation for the symmetry measure. Uniform distribution

function [s] = sym_measure (matrix)
        
    upper = triu(matrix,1);         %extract the upper triangle matrix
    lower = tril(matrix,-1)';       %extract the lower triangle matrix and transpose it

    x = upper(:);                   %convert the matrix into a vector
    y = lower(:);                   %convert the matrix into a vector

    temp = x + y;                   %sum vector elements==sum the reciprocal elements of the matrix
    nonzero_index = find(temp~=0.); %create a vector whoose elements are the index of the non zero elements in temp
    K = length(nonzero_index);      %counts how many elements of temp are nonzero==counts the number of pairs connections for which at least one direction is nonzero

    if K > 0
        s = 1 - sum ( abs(x(nonzero_index)-y(nonzero_index)) ./ (x(nonzero_index)+y(nonzero_index)) ) / K;
    else
        s = 0;
    end
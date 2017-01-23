function H = entropy_true(p)
%entropy_true  computes Shannon entropy H(p) in bits for the input discrete
%              distribution.
%
% This function returns a scalar entropy when the input distribution p is a 
% vector of probability masses, or returns in a row vector the columnwise 
% entropies of the input probability matrix p.

% Error-check of the input distribution
p0 = p(:);
if any(imag(p0)) || any(isinf(p0)) || any(isnan(p0)) || any(p0<0) || any(p0>1)
    error('The probability elements must be real numbers between 0 and 1.');
elseif any(abs(sum(p)-1) > sqrt(eps))
    error('Sum of the probability elements must equal 1.');
end

H = -sum(log2(p.^p));  
end
function MI = MI_true(Pxy)
%MI_true  computes the mutual information for the input joint distribution
% 
% Input:
% ----- Pxy: Sx-by-Sy matrix describing the joint probability masses of the
%            the bivariate (X,Y), where Sx and Sy are the alphabet size of 
%            X and Y, respectively. The (i,j)-th entry of Pxy denotes the
%            joint probability Pr(X = i,Y = j).
%
% Output:
% ----- MI: the scalar output of the mutual information I(X;Y). 
   
	% Error-check of the input distribution 
    pxy = Pxy(:);
    if any(imag(pxy)) || any(isinf(pxy)) || any(isnan(pxy)) || any(pxy<0) || any(pxy>1)
        error('The probability elements must be real numbers between 0 and 1.');
    elseif abs(sum(pxy)-1) > sqrt(eps)
        error('Sum of the probability elements must equal 1.');
    end

    % Calculate the marginals of X and Y
    px = sum(Pxy,2);   
    py = sum(Pxy,1);   

    % I(X;Y) = H(X) + H(Y) - H(X,Y)
    MI = entropy_true(px) + entropy_true(py) - entropy_true(pxy);
end
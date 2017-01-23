function est = est_entro_JVHW(samp)
%est_entro_JVHW   Proposed JVHW estimate of Shannon entropy (in bits) of 
%                 the input sample
%
% This function returns a scalar JVHW estimate of the entropy of samp when 
% samp is a vector, or returns a row vector containing the JVHW estimate of
% each column of samp when samp is a matrix.
%
% Input:
% ----- samp: a vector or matrix which can only contain integers. The input
%             data type can be any interger classes such as uint8/int8/
%             uint16/int16/uint32/int32/uint64/int64, or floating-point 
%             such as single/double. 
% Output: the entropy (in bits) of the input vector or that of each column 
%         of the input matrix. The output data type is double.


    if ~isequal(samp, fix(samp))
        error('Input sample must only contain integers.');
    end

    if isrow(samp)
        samp = samp.';
    end
    [n, wid] = size(samp);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    order = min(4+ceil(1.2*log(n)), 22);  % The order of polynomial is no more than 22 because otherwise floating-point error occurs
    persistent poly_entro;
    if isempty(poly_entro)
        load poly_coeff_entro.mat poly_entro;
    end
    coeff = poly_entro{order};   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     % A fast algorithm to compute the column-wise histogram of histogram (fingerprint) 
%     f = find([diff(sort(samp))>0; true(1,wid)]);  % Returns in column vector f the linear indices to the last occurrence of repeated values along every column of samp
%     f = accumarray({filter([1;-1],1,f),ceil(f/n)},1)   % f: fingerprint   

    % A memory-efficient algorithm for computing fingerprint when wid is large, e.g., wid = 100
    d = [true(1,wid);logical(diff(sort(samp),1,1));true(1,wid)];
    for k = wid:-1:1
        a = diff(find(d(:,k)));
        id = 1:max(a);  
        f(id,k) = histc(a,id);
    end

    prob = (1:size(f,1))/n;
        
    % Piecewise linear/quadratic fit of c_1
    V1 = [0.3303 0.4679];     
    V2 = [-0.530556484842359,1.09787328176926,0.184831781602259];   
    f1nonzero = f(1,:) > 0;
    c_1 = zeros(1, wid);
    if n >= order && any(f1nonzero)
        if n < 200
            c_1(f1nonzero) = polyval(V1, log(n./f(1,f1nonzero)));   
        else
            n2f1_small = f1nonzero & log(n./f(1,:)) <= 1.5;
            n2f1_large = f1nonzero & log(n./f(1,:)) > 1.5;
            c_1(n2f1_small) = polyval(V2, log(n./f(1,n2f1_small)));  
            c_1(n2f1_large) = polyval(V1, log(n./f(1,n2f1_large)));  
        end
        c_1(f1nonzero) = max(c_1(f1nonzero), 1/(1.9*log(n)));  % make sure nonzero threshold is higher than 1/n
    end
    
    prob_mat = entro_mat(prob, n, coeff, c_1);
    est = sum(f.*prob_mat, 1)/log(2); 
end 


function output = entro_mat(x, n, g_coeff, c_1)
    K = length(g_coeff) - 1;   % g_coeff = {g0, g1, g2, ..., g_K}, K: the order of best polynomial approximation, 
    thres = 4*c_1*log(n)/n;
    [T, X] = meshgrid(thres,x);   
    ratio = min(max(2*X./T-1,0),1);
    q = reshape(0:K-1,1,1,K); 
    g = reshape(g_coeff,1,1,K+1);  
    MLE = -X.*log(X) + 1/(2*n);   
    polyApp = sum(bsxfun(@times, cumprod(cat(3, T, bsxfun(@minus, n*X, q)./bsxfun(@times, T, n-q)),3), g), 3) - X.*log(T);     
    polyfail = isnan(polyApp) | isinf(polyApp); 
    polyApp(polyfail) = MLE(polyfail);   
    output = ratio.*MLE + (1-ratio).*polyApp; 
    output = max(output,0);
end
 

 

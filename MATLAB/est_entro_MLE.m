function est = est_entro_MLE(samp)
%est_entro_MLE  Maximum likelihood estimate of Shannon entropy (in bits) of 
%               the input sample
%
% This function returns a scalar MLE of the entropy of samp when samp is a 
% vector, or returns a (row-) vector consisting of the MLE of the entropy 
% of each column of samp when samp is a matrix.
%
% Input:
% ----- samp: a vector or matrix which can only contain integers. The input
%             data type can be any interger classes such as uint8/int8/
%             uint16/int16/uint32/int32/uint64/int64, or floating-point 
%             such as single/double. 
% Output:
% ----- est: the entropy (in bits) of the input vector or that of each 
%            column of the input matrix. The output data type is double. 


    if ~isequal(samp, fix(samp))
        error('Input sample must only contain integers.');
    end

    if isrow(samp)
        samp = samp.';
    end
    [n, wid] = size(samp);

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
    prob_mat = -prob.*log2(prob);
    est = prob_mat * f;
end
function est = est_MI_MLE(X,Y)
% est = est_MI_MLE(X,Y) 
%
% This function returns the scalar MLE of the mutual information I(X;Y)
% when both X and Y are vectors, and returns a row vector consisting 
% of the estimate of mutual information between each corresponding column 
% of X and Y when they are matrices. 
%
% Input: 
% ----- X, Y: two vectors or matrices with the same size, which can only
%             contain integers. 
%
% Output: 
% ----- est: the estimate of the mutual information (in bits) between input 
%            vectors or that between each corresponding column of the input 
%            matrices. The output data type is double. 
    
    if any(size(X) ~= size(Y))
        error('Input arguments X and Y should be of the same size!');
    end
    
	if ~isequal(X, fix(X)) || ~isequal(Y, fix(Y))
        error('Input arguments X and Y must only contain integers!');
    end
    
	if isrow(X)
        X = X.';
        Y = Y.';
    end
    szX = size(X); szY = size(Y);
    
    if ~isequal(class(X), class(Y), 'int64')
        X = int64(X);
        Y = int64(Y);
    end

    [X, id] = sort(X); 
    X(bsxfun(@plus, id, (0:szX(2)-1)*szX(1))) = cumsum([ones(1,szX(2));diff(X)>0]);
    [Y, id] = sort(Y);
    Y(bsxfun(@plus, id, (0:szY(2)-1)*szY(1))) = cumsum([ones(1,szY(2));diff(Y)>0]);
    
    % I(X,Y) = H(X) + H(Y) - H(X,Y)
    est = max(0, est_entro_MLE(X) + est_entro_MLE(Y) - est_entro_MLE(bsxfun(@times, X-1, max(Y))+Y)); 
end







function est = est_MI_JVHW(X,Y)
% est = est_MI_JVHW(X,Y) 
%
% This function returns our scalar estimate of mutual information I(X;Y)
% when both X and Y are vectors, and returns a row vector consisting 
% of the estimate of mutual information between each corresponding column 
% of X and Y when they are matrices. 
%
% Input: 
% ----- X, Y: two vectors or matrices with the same size, which can only
%             contain integers. 
%
% Output: 
% ----- est: the estimate of the mutual information between input vectors
%            or that between each corresponding column of the input 
%            matrices. The output data type is double. 
    
    if any(size(X) ~= size(Y))
        error('Input arguments X and Y should be of the same size.');
    end
    
	if ~isequal(X, fix(X)) || ~isequal(Y, fix(Y))
        error('Input arguments X and Y must only contain integers.');
    end
    
	if isrow(X)
        X = X.';
        Y = Y.';
    end
    szX = size(X); szY = size(Y);
    
    % For entropy estimation with S = 1e7, the input samp only takes values 
    % in [1,S]; thus, int32 suffices to represent all possible integers.
    % For mutual information estimation, however, the bivariate pair (X,Y) 
    % takes theoretically S^2 = 1e14 distinct values; thus, we use int64 to
    % represent all possibilities. As a reference, the largest integer 
    % values various data types can represent in MATLAB are summarized in 
    % the following table. Note that the integer range of int64 is much 
    % is much larger than that of double.
    %  ====================================================================
    %  |    Data Type   |   The largest integer value can be represented  |
    %  ====================================================================
    %  |     int32      |     2^31 - 1 = 2.147483647000000e+09            |
    %  |     int64      |     2^63 - 1 = 9.223372036854776e+18            |
    %  |     single     |     flintmax('single') = 1.6777216e+07          |
    %  |     double     |     flintmax('double') = 9.007199254740992e+15  |
    %  ====================================================================
    if ~isequal(class(X), class(Y), 'int64')
        X = int64(X);
        Y = int64(Y);
    end
    
    % Map integer data along each column of X and Y to consecutive integer
    % numbers (which start with 1 and end with the total number of distinct 
    % values in each corresponding column). For example, 
    %                 [  1    6    4  ]        [ 1  3  3 ]                  
    %                 [  2    6    3  ] -----> [ 2  3  2 ]
    %                 [  3    2    2  ]        [ 3  1  1 ]
    %                 [ 1e5   3   100 ]        [ 4  2  4 ]
    % The purpose of this data mapping is to make the effective data range
    % as small as possible, minimizing the possibility of overflows.
    [X, id] = sort(X); 
    X(bsxfun(@plus, id, (0:szX(2)-1)*szX(1))) = cumsum([ones(1,szX(2));diff(X)>0]);
    [Y, id] = sort(Y);
    Y(bsxfun(@plus, id, (0:szY(2)-1)*szY(1))) = cumsum([ones(1,szY(2));diff(Y)>0]);

    % I(X,Y) = H(X) + H(Y) - H(X,Y)
    est = max(0, est_entro_JVHW(X) + est_entro_JVHW(Y) - est_entro_JVHW(bsxfun(@times, X-1, max(Y))+Y)); 
end







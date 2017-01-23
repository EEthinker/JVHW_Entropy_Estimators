function x = randsmpl(p, m, n, varargin)
%RANDSMPL  Independent sampling from a discrete distribution.
%          
%   x = randsmpl(p, m, n) returns an m-by-n matrix x with random samples
%   drawn independently from the input (discrete) distribution specified
%   with pmf p. Suppose that the sample space comprises K samples, then p
%   must be a (row- or column-) vector containing K probability masses 
%   summing to 1. The output, x(i,j) = k, k = 1, ..., K, describes that
%   the k-th sample is drawn in the (i,j)-th trial, for i = 1, ..., m and
%   j = 1,...,n. The default output data type of x is 'double'.
%
%   x = randsmpl(p, m, n, classname) returns an m-by-n matrix x whose data  
%   type is specified by classname. The classname must be a valid numeric 
%   class name which includes the following
%     'double' (default) | 'single' | 'int64'  | 'int32' | 'int16' | ...
%      'int8' | 'uint64' | 'uint32' | 'uint16' | 'uint8' | 
%   If classname is not provided, the output type is default to 'double'. 
%
%   Remarks:
%       - The main idea is to divide interval [0,1] into K disjoint bins, 
%         each with length proportional to the corresponding probability
%         mass. Then, we draw samples from the unfiorm distribution U(0,1)  
%         and determine the indices of the bins containing those samples.    
%
%       - histc and histcounts (introduced in R2014b) determine not only  
%         the indices of the bins, but also the histogram counts within 
%         each bin. In contrast, discretize (introduced in R2015a) is aimed
%         solely for finding the indices of the histogram bins, thus it is
%         much faster than histc and histcounts. We also observed that
%         discretize is considerably more efficient in terms of memory 
%         consumption than histc. Thus, for both memeory and performance
%         considerations, we choose to use discretize.
%
%       - Two alternatives are provided for backward compatibility with 
%         earlier MATLAB releases, e.g., R2014b or earlier:
%         1) discretizemex (already included in the folder): the binary mex 
%            file of discretize;
%         2) Use interp1 instead: though slightly slower than discretize, 
%            the performance penalty is negligible.
%
%   See also RAND, RANDI, RANDN, RNG.
%
%   Peng Liu, Nov. 18, 2015

narginchk(3, 4);
if nargin < 4
    classname = 'double'; % Consistent with MATLAB's default double-precision computation
else
    classname = varargin{:};
    if ~ismember(classname,{'int8','int16','int32','int64','uint8','uint16','uint32','uint64','single','double'})
        error('CLASSNAME input must be a valid numeric class name, for example, ''int32'' or ''double''.');
    end  
end

if ~isvector(p)
    error('Input distribution p must be a vector.')
end

% Error-check of the input distribution
if any(imag(p)) || any(isinf(p)) || any(isnan(p)) || any(p < 0) || any(p > 1)
    error('The probability elements must be real numbers between 0 and 1.');
elseif abs(sum(p) - 1) > sqrt(eps)
    error('Sum of the probability elements must equal 1.');
end

edges = [0; cumsum(p(:))];
if abs(edges(end) - 1) > sqrt(eps)   % Deal with floating-point errors due to cumulative sum
    edges = edges/edges(end);
end
edges(end) = 1 + eps(1);

x = cast(discretizemex(rand(m, n), edges),classname);        % For R2014b or earlier releases
% x = cast(discretize(rand(m, n), edges),classname);         % For R2015a or later releases
% x = cast(interp1(edges,1:length(edges),rand(m, n),'previous'),classname);   % Alternative method if discretize/discretizemex don't work

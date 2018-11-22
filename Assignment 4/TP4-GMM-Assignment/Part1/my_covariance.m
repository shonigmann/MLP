function [ Sigma ] = my_covariance( X, X_bar, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o X_bar : (N x 1), an Nx1 matrix corresponding to mean of data X
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%

% Auxiliary Variable
[N, M] = size(X);

% Output Variable
Sigma = zeros(N, N);

if strcmp(type,'full')
    
    X = X-X_bar; %check this line
    Sigma = 1/(M-1)*(X*X');
    
elseif strcmp(type,'diag')
    
    X = X-X_bar; %check this line
    Sigma = 1/(M-1)*(X*X');
    Sigma = diag(diag(Sigma));
    
elseif strcmp(type,'iso')
    X = X-X_bar;
    sigmaSquared =0;
    for m=1:M
        sigmaSquared = sigmaSquared + 1/(N*M)*norm(X(:,m)).^2;
    end
    Sigma = eye(N)*sigmaSquared;
    
end


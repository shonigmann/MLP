function [Priors,Mu,Sigma] = maximization_step(X, Pk_x, params)
    %MAXIMISATION_STEP Compute the maximization step of the EM algorithm
    %   input------------------------------------------------------------------
    %       o X         : (N x M), a data set with M samples each being of 
    %       o Pk_x      : (K, M) a KxM matrix containing the posterior probabilty
    %                     that a k Gaussian is responsible for generating a point
    %                     m in the dataset, output of the expectation step
    %       o params    : The hyperparameters structure that contains k, the number of Gaussians
    %                     and cov_type the coviariance type
    %   output ----------------------------------------------------------------
    %       o Priors    : (1 x K), the set of updated priors (or mixing weights) for each
    %                           k-th Gaussian component
    %       o Mu        : (N x K), an NxK matrix corresponding to the updated centroids 
    %                           mu = {mu^1,...mu^K}
    %       o Sigma     : (N x N x K), an NxNxK matrix corresponding to the
    %                   updated Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
    %%

    % Additional variables
    [N, M] = size(X);
    Priors = zeros(1,params.k);
    Mu = zeros(N,params.k);
    Sigma = zeros(N,N,params.k);
    eps = 1e-5;
    K = params.k;

    prob_sum = sum(Pk_x,2); %recurring bit in many expressions. no use processing multiple times

    for k=1:K
        Priors(k) = 1/M*prob_sum(k); %eq9    
        for m=1:M
            Mu(:,k) = Mu(:,k) + Pk_x(k,m)*X(:,m)/(prob_sum(k)); %eq10
        end
    end

    if strcmp(params.cov_type,'full')
        for k=1:K
            for m=1:M
                Sigma(:,:,k) = Sigma(:,:,k) + (Pk_x(k,m)*(X(:,m)-Mu(:,k))*(X(:,m)-Mu(:,k))')/prob_sum(k); %eq11
            end
        end
        %adding tiny variance to prevent numerical instability
        for k=1:K
            Sigma(:,:,k) = squeeze(Sigma(:,:,k)) + eps*eye(N);
        end
    elseif strcmp(params.cov_type,'diag')
        for k=1:K
            for m=1:M
                Sigma(:,:,k) = Sigma(:,:,k) + (Pk_x(k,m)*(X(:,m)-Mu(:,k))*(X(:,m)-Mu(:,k))')/prob_sum(k); %eq11    
            end
            Sigma_temp = zeros(N);
            for i=1:N 
                Sigma_temp(i,i) = Sigma(i,i,k);%extract only diagonal elements
            end
            Sigma(:,:,k) = Sigma_temp;
        end
        %adding tiny variance to prevent numerical instability
        for k=1:K
            Sigma(:,:,k) = squeeze(Sigma(:,:,k)) + eps*eye(N);
        end

    elseif strcmp(params.cov_type,'iso')
        %eq12
        for k=1:K
            for m=1:M
                Sigma(:,:,k) = Sigma(:,:,k) + eye(N)*(Pk_x(k,m)*(norm(X(:,m)-Mu(:,k)).^2))/(N*prob_sum(k)); %eq11
            end
        end
        
        %adding tiny variance to prevent numerical instability
        for k=1:K
            Sigma(:,:,k) = squeeze(Sigma(:,:,k)) + eps*eye(N);
        end
    end
   
end

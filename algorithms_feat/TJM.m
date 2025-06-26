function X_new = TJM(X, domain_label, Y, param)

% Transfer Joint Matching (TJM)
% Reference: Mingsheng Long. Transfer Joing Matching for visual domain adaptation. CVPR 2014.

% Inputs:
%%% X                 :   All samples in source and target domain. n-by-m matrix, n: sample number, m: feature dimension
%%% domain_label :	n-by-1 logical vector, domain_label(i) = 1 (true): source sample; 0 (false) : target sample.
%%% Y                 :   labels for all the samples
%%% param         :     option struct
%%%%% lambda    :     regularization parameter
%%%%% dim          :     dimension after adaptation, dim <= n_feature
%%%%% kernel_tpye  :     kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%% gamma        :     bandwidth for rbf kernel, can be missed for other kernels
%%%%% T            :     n_iterations, T >= 1. T <= 10 is suffice

% Outputs:
%%% X_new            :   All the samples after adaptation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==============>> Set your parameters here !!! <<====================
param.dim = 2;  
param.lambda = 0.1; 
param.kernel_type = 'rbf'; 
param.gamma = 0.1; 
param.T = 10;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	lambda = param.lambda;              %% lambda for the regularization
	dim = param.dim;                    %% dim is the dimension after adaptation, dim <= m
	kernel_type = param.kernel_type;    %% kernel_type is the kernel name, primal|linear|rbf
	gamma = param.gamma;                %% gamma is the bandwidth of rbf kernel
	T = param.T;                        %% iteration number
    
	%fprintf('TJM: dim=%d  lambda=%f\n',dim,lambda);
    X_src = X(domain_label,:);
    X_tar = X(~domain_label,:);
    Y_src = Y(domain_label,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Set predefined variables
	X = [X_src',X_tar'];
	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	ns = size(X_src,1);
	nt = size(X_tar,1);
	n = ns+nt;

	% Construct kernel matrix
	K = kernel_tjm(kernel_type,X,[],gamma);

	% Construct centering matrix
	H = eye(n)-1/(n)*ones(n,n);

	% Construct MMD matrix
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    C = length(unique(Y_src));
	M = e*e' * C;
	
    Cls = [];
	% Transfer Joint Matching: JTM
	G = speye(n);
	for t = 1:T
        %%% Mc [If want to add conditional distribution]
        N = 0;
        if ~isempty(Cls) && length(Cls)==nt
            for c = reshape(unique(Y_src),1,C)
                e = zeros(n,1);
                e(Y_src==c) = 1 / length(find(Y_src==c));
                e(ns+find(Cls==c)) = -1 / length(find(Cls==c));
                e(isinf(e)) = 0;
                N = N + e*e';
            end
        end
        M = M + N;
        
        M = M/norm(M,'fro');
        
	    [A,~] = eigs(K*M*K'+lambda*G,K*H*K',dim,'SM');
	%     [A,~] = eigs(X*M*X'+lambda*G,X*H*X',k,'SM');
	    G(1:ns,1:ns) = diag(sparse(1./(sqrt(sum(A(1:ns,:).^2,2)+eps))));
	    Z = A'*K;
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
	    Zs = Z(:,1:ns)';
	    Zt = Z(:,ns+1:n)';
        
        X_new = [Zs; Zt];
    end
end


% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix

function K = kernel_tjm(ker,X,X2,gamma)

switch ker
    case 'linear'
        
        if isempty(X2)
            K = X'*X;
        else
            K = X'*X2;
        end

    case 'rbf'

        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-gamma*D); 

    case 'sam'
            
        if isempty(X2)
            D = X'*X;
        else
            D = X'*X2;
        end
        K = exp(-gamma*acos(D).^2);

    otherwise
        error(['Unsupported kernel ' ker])
end
end
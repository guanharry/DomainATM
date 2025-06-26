function X_new = ITL(X, domain_label, Y, param)

% Information-Theoretical Learning (ITL) 
% Reference: Y. Shi and F. Sha, "Information-theoretical learning of discriminative
% clusters for unsupervised domain adaptation," in ICML, 2012.
%
% inputs:
%%% X                    : All samples in source and target domains. n-by-m matrix, n is the number of samples, m is the dimension of features.
%%% domain_label :	n-by-1 logical vector, domain_label(i) = 1 (true): source sample; 0 (false) : target sample.
%%% Y                     :   labels for all the samples
%%% param             :   option struct

% outputs:
%%% X_new: all the samples after adaptation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==============>> Set your parameters here !!! <<====================
param.dim = 2; 
param.lambda = 10;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set parameters
pcaCoef = param.dim;  %
lambda = param.lambda;     % regularization parameter


Xs = X(domain_label,:);
Xt = X(~domain_label,:);

Ys = Y(domain_label,:);

%% compute
[~,pcaModelS] = PCA(Xs,[],struct('pcaCoef',pcaCoef));
[~,pcaModelT] = PCA(Xt,[],struct('pcaCoef',pcaCoef));

d = min(size(pcaModelS.W_prj,2),size(pcaModelT.W_prj,2));

W_prjT = pcaModelT.W_prj(:,1:d);

L = infometric(W_prjT, Xs, Ys, Xt, lambda);

%% project the samples
X_new = zeros(size(X,1), d);
X_new(domain_label,:) = Xs * L;
X_new(~domain_label,:) = Xt * L;

end
function X_new = TCA(X, domain_label, Y, param)

% Transfer Component Analysis (TCA)
% ref: S. J. Pan, I. W. Tsang, J. T. Kwok, and Q. Yang, "Domain adaptation
%	    via transfer component analysis," Neural Networks, IEEE Trans, 2011
% 
% Inputs:
%%% X: All samples in source and target domain. n-by-m matrix, n: sample
%%%      number, m: feature dimension

%%% domain_label:	n-by-1 logical vector, domain_label(i) = 1 (true): source
%%%                          sample; 0 (false) : target sample.
% %%
%%%Y: category label for all the samples. Useless in this algorithm
%%%
%%% param: Struct of hyper-parameters
%%%% param.kerName:  kernel name: 'lin', 'rbf', 'poly', 'lap'
%%%% param.mu: the weight of the regularization term, see the ref
%%%% param.dim: the dimension of the subspace
%%%% param. kerSigma: kernel parameter

% output:
%%%% X_new: All the samples after adaptation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==============>> Set your parameters here !!! <<====================
param.dim = 2; 
param.kerName = 'lin'; 
param.kerSigma = 1; 
param.mu = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isRegress = 0; % 0 for classification problem, 1 for regression
kerName = param.kerName; % kernel name: 'lin', 'rbf', 'poly', 'lap'
mu = param.mu; % the weight of the regularization term, see the ref
m = param.dim;  % the dimension of the subspace

kerSigma = param. kerSigma; % kernel parameter, see the next cell ("kernels")
bSstca = false; % 0 for TCA if no label information is considered for all 
	% samples, 1 for semisupervised TCA (SSTCA) if some labels are
	% considered.
doSample = false; % when there are too many unlabeled data, eigenvalue 
	% decomposition can be slow. Setting this variable to true makes the
	% code to sample some unlabeled data.
nSmpRatio = 1; % if doSample=true, the number of unlabeled data to sample 
	% will be ceil(ntr*nSmpRatio)

% SSTCA params
lambda = 1; % weight of the geometry term
knn = 5; % #neighbor when computing the Laplacian matrix
geoSigma = 0.01;
gamma = .1; % the weight of the supervised term, only useful in SMIDA, see 
	% the ref

%% kernels
nm = @(X,p)repmat(sum(X.^2,2),1,p);
linKer = @(X1,X2)X1*X2';
rbfKer = @(X1,X2)exp(-(nm(X1,size(X2,1))+nm(X2,size(X1,1))'-2*X1*X2')/2/kerSigma^2);
polyKer = @(X1,X2)(1+kerSigma*X1*X2').^2;
lapKer = @(X1,X2)exp(-pdist2(X1,X2)/kerSigma);
if strcmpi(kerName,'lin'), kerFun = linKer;
elseif strcmpi(kerName,'poly'), kerFun = polyKer;
elseif strcmpi(kerName,'rbf'), kerFun = rbfKer;
elseif strcmpi(kerName,'lap'), kerFun = lapKer;
else error('unknown ker'); end

%% sort samples
ftLabeled = X(domain_label,:);
ftUnlabeled = X(~domain_label,:);
Xs = X(domain_label,:);
Xt = X(~domain_label,:);
nl = size(ftLabeled,1);
nul = size(ftUnlabeled,1);
nsrc = size(Xs,1);
ntar = size(Xt,1);
if doSample % sample some target domain data
	rng(0)
    ntarNew = min(ntar,floor(nsrc*nSmpRatio));
    ftAll1 = [Xs; Xt(randperm(ntar,ntarNew),:)];
	ntar = ntarNew;
else
    ftAll1 = [Xs; Xt];
end

%% compute
K = kerFun(ftAll1,ftAll1);
L = [ones(nsrc)/nsrc^2, -ones(nsrc,ntar)/nsrc/ntar;
    -ones(ntar,nsrc)/nsrc/ntar,ones(ntar)/ntar^2];
H = eye(nsrc+ntar)-ones(nsrc+ntar)/(nsrc+ntar);

if ~bSstca % TCA
    A = (K*L*K+mu*eye(nsrc+ntar))\K*H*K;
else % SSTCA
    % laplacian matrix
    M = squareform(pdist(ftAll1));
	geoSigma = mean(pdist(ftAll1));
	M = exp(-M.^2/2/geoSigma^2);
	M = M-eye(nsrc+ntar);
	Msort1 = sort(M,'descend');
    for p = 1:nsrc+ntar
        M(M(:,p)<Msort1(knn,p),p) = 0; % k near neighbors
    end
    M = max(M,M');
	D = diag(sum(M,2));
	Lgeo = D-M;

    if isRegress==0 && all(target==floor(target)) % clsf
        Kyy = repmat(target,1,nl)==repmat(target,1,nl)';
    else % regress
        Kyy = target*target'; % lin ker
    end
    Kyy_tilde = (1-gamma)*eye(nsrc+ntar);
	
	maLabeledNew = domain_label; 
	
    Kyy_tilde(maLabeledNew,maLabeledNew) = ...
		Kyy_tilde(maLabeledNew,maLabeledNew)+gamma*Kyy;
	
    A = (K*(L+lambda*Lgeo)*K+mu*eye(nsrc+ntar))\...
        K*H*Kyy_tilde*H*K;
end

maBad = isnan(A)|isinf(A);
if any(any(maBad)), A(maBad)=0; end
[V,D] = eig(A);
[D,I] = sort(diag(D),'descend');
transMdl.W = real(V(:,I(1:m)));

%% project the samples
KNew = kerFun(X, ftAll1);
X_new = KNew*transMdl.W;

end

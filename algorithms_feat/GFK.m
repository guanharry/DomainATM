function X_new = GFK(X, domain_label, Y, param)
% Geodesic Flow Kernel (GFK) 
% Reference: Gong et.al. Geodesic Flow Kernel for Unsupervised Domain Adaptation, CVPR 2012
% 
% inputs:
%%% X                    : All samples in source and target domains. n-by-m matrix, n is the number of samples, m is the dimension of features.
%%% domain_label :	n-by-1 logical vector, domain_label(i) = 1 (true): source sample; 0 (false) : target sample.
%%% Y                     :   labels for all the samples
%%% param             :   option struct
%%%%% dim             :    ratio that controls the dimension of the subspace. If 0, will be automatically computed
%%%%% bYs           :    if use the label info in source domain, 0: not use; 1: use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==============>> Set your parameters here !!! <<====================

param.dim = 0; 
param.bYs = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
dr = param.dim; 
bYs = param.bYs; 


X(domain_label,:) = zscore(X(domain_label,:)); % according to gfk's sample code
X(~domain_label,:) = zscore(X(~domain_label,:));

nFt = size(X, 2);
Xs = X(domain_label,:);
Xt = X(~domain_label,:);

Ys = Y(domain_label,:);
%% compute
if ~bYs
	Ps = pca(Xs);  % source subspace
else
	addpath plslda
	model = pls_basis(Xs, Ys, min(size(Xs)),'none');
	Ps = model.weight;
end
Pt = pca(Xt);  % target subspace
Pst = pca(X);

%% select subspace dimension according to ref 1
maxd = min(cellfun(@(x)size(x,2),{Ps,Pt,Pst}));
Ps = Ps(:,1:maxd); 
Pt = Pt(:,1:maxd);
Pst = Pst(:,1:maxd);
PsPst = abs(diag(Ps'*Pst)); % abs?
PtPst = abs(diag(Pt'*Pst));
PsLen = sqrt(sum(Ps.^2,1))'; % should be all 1
PtLen = sqrt(sum(Pt.^2,1))';
PstLen = sqrt(sum(Pst.^2,1))';
alphas = acos(PsPst./PsLen./PstLen);
betas = acos(PtPst./PtLen./PstLen);
Dd = (sin(alphas)+sin(betas))/2;
d = find(Dd>1-1e-3,1,'first');
if isempty(d), d = maxd; end
d = min(d,floor(nFt/2));

if dr ~= 0 % manually set the subspace dimension
	d = floor(min(maxd,floor(nFt/2))*dr);
end

%% compute
Ps = Ps(:,1:d);
Pt = Pt(:,1:d);
G = GFK_core([Ps,null(Ps')], Pt);
[TL,TD]=ldl(G);
L=TL*(TD.^0.5);

transMdl.W = real(L(:,1:2*d)); % imaginary after d*2 because rank deficient

%% project the samples
X_new = X*transMdl.W;

end

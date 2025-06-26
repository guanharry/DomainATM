function X_new =  SA(X, domain_label, Y, param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Subspace Alignment (SA)
%  ref: B. Fernando, et.al "Unsupervised visual domain adaptation using 
%        subspace alignment," in ICCV, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% input
%%% X: All samples in source and target domain. n-by-m matrix, n: sample number, m: feature dimension
%%% domain_label:	n-by-1 logical vector, domain_label(i) = 1 (true): source sample; 0 (false) : target sample.
%%% Y: Category labels for all the samples, useless in this algorithm
%%% param: Struct of hyper-parameters for the algorithm
%%%%%param.dim: the dimension after adaptation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%% X_new:	All the samples after adaptation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==============>> Set your parameters here !!! <<======================
param.dim = 2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%% 
Xs = X(domain_label,:);    
Xt = X(~domain_label,:);

[Ps, Xs_new, ~] = pca(Xs);
[Pt, Xt_new, ~] = pca(Xt);

Ps = Ps(:,1 : param.dim );
Pt  = Pt(:,1 : param.dim );

Xs_new =  Xs_new(:, 1: param.dim); 
Xs_new = Xs_new * Ps' * Pt;

Xt_new =  Xt_new(:, 1: param.dim);  %Xt *Pt;

X_new = [Xs_new; Xt_new];

end


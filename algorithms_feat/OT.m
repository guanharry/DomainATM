function X_new = OT(X, domain_label, Y, param)

%  Optimal Transport
%  ref: Guan, et.al. Multi-source domain adaptation via optimal transport for brain dementia identification. 
%In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)

% Inputs:
%%% X: All samples in source and target domain. n-by-m matrix, n: sample
%%%      number, m: feature dimension

%%% Y: Category labels for all the samples. Useless in this algorithm

%%% domain_label:	n-by-1 logical vector, domain_label(i) = 1 (true): source
%%%                          sample; 0 (false) : target sample.
% %%
%%% param: Struct of hyper-parameters
%%%% param.reg: the regularization term for the sinkhorn algorithm

% output:
%%% X_new: All the samples after adaptation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==============>> Set your parameters here !!! <<====================
param.reg = 0.01;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	reg = param.reg;                  
          
    Xs = X(domain_label,:);
    Xt = X(~domain_label,:);
    
    Xs_new = sinkhornDa(Xs, Xt, reg);
    
    X_new = [Xs_new; Xt];
    
end
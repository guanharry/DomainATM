
function X_new = CORAL(X, domain_label, Y, param)
%  Correlation Alignment (CORAL)
%  ref: Sun, et.al . "Return of frustratingly easy domain adaptation." 
%        Proceedings of the AAAI Conference on Artificial Intelligence. 2016.

% Inputs:
%%% X                    : All samples in source and target domain. n-by-m matrix, n: sample
%%%                          number, m: feature dimension

%%% domain_label:	n-by-1 logical vector, domain_label(i) = 1 (true): source
%%%                           sample; 0 (false) : target sample.

%%% Y:	category label, useless in this algorithm
% 
%%% param: Struct of hyper-parameters, useless in this algorithm


% output:
%%% X_new: All the samples after adaptation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%==============>> Set your parameters here !!! <<====================
% There is no parameter needed for CORAL :)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% source and target samples
Xs = X(domain_label,:);
Xt = X(~domain_label,:);
    
cov_src = cov(Xs) + eye(size(Xs,2));
cov_tar = cov(Xt) + eye(size(Xt,2));
A_coral = cov_src^(-1/2) * cov_tar^(1/2);

Xs_new = Xs * A_coral;
       
X_new = [Xs_new; Xt];

end

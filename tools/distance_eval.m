function d = distance_eval(X, domain_label)
% 
%	X: all input samples 
%	domain_label: domain labels （logical value (Source:1 (true), target: 0 (false))
%%
%  d: MMD distance


%% sort samples
Xs = X(domain_label,:);
Xt = X(~domain_label,:);

d = mmd(Xs, Xt, 'rbf');

end
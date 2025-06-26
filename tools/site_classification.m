

function acc = site_classification(X, Y, domain_label)

addpath('./algorithms')
rng('default') % For reproducibility

Xs = X(domain_label, :);  Ys = Y(domain_label, :); 
Xt = X(~domain_label, :);  Yt = Y(~domain_label, :); 


Ys_site = ones(size(Xs, 1),1);
Yt_site = zeros(size(Xt, 1), 1);
Y_site = [Ys_site; Yt_site];

randIndex = randperm(size(X, 1));

X_r = X(randIndex,:);
Y_site_r = Y_site(randIndex, :);

X_train = X_r(1: 0.6*size(X_r, 1)-1,:);
Y_train = Y_site_r(1: 0.6*size(Y_site_r, 1)-1,:);

X_test = X_r(0.6*size(X_r, 1) : end, :);
Y_test = Y_site_r(0.6*size(Y_site_r, 1) : end, :);

Model =  fitcknn(X_train, Y_train, 'NumNeighbors',5);

[ypred, score] = predict(Model, X_test);
prob = max(score, [], 2);
[acc, auc, sen, spe, bac] = result_analysis(ypred, prob, Y_test);

end
function transformed_sample = sinkhornDa(Xs, Xt, reg, options)
    if ~exist('reg','var')
     % default the regularization parameter
      reg = 0.1;
    end
    if ~exist('options', 'var')
        options.niter = 5000; % number of iterations
        options.verb = 0;
        options.rho = Inf; % used balanced transport
        options.tau = 0;
        options.metric = 'squaredeuclidean';
    end
    Ns = length(Xs);
    Nt = length(Xt);
    %for our purposes samples are uniform
    mu = ones(Ns,1)/Ns;
    nu = ones(Nt,1)/Nt;
    
    
    %compute the cost function In this case squared euc
    %distmat = @(x,y)sum(x.^2,1)' + sum(y.^2,1) - 2*x.'*y;
    %c3 = distmat(Xs', Xt');
    %x2 = sum(Xs.^2,1); y2 = sum(Xt.^2,1);
    c = pdist2(Xs, Xt, 'squaredeuclidean');
    
    %run the algo
 
    [u,v,gamma,Wprimal,Wdual,err] = sinkhorn_log(mu,nu,c,reg,options);
    
    %Gamma is coupling so just compute as is
    transp_sum = sum(gamma, 2);
    transp = gamma ./ transp_sum;
    transported_Xs = transp * Xt;
    
    
    transformed_sample = transported_Xs;
end
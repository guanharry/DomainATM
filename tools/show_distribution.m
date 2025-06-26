function show_distribution(X, Y, domain_label, axlb, name, metric)

% figure
hold on, 

S_y = Y(domain_label,:);            
T_y = Y(~domain_label,:);
            
numPos_1 = length(find(S_y==1));
            
numNeg_1 = length(find(S_y==2));
            
numPos_2 = length(find(T_y==1));
            
numNeg_2 = length(find(T_y==2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
XX = X; %XX is used to display X

if(size(XX,2)>2)
    XX = tsne(XX);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c1 = [1, 0, 0; 1, 0, 0];
c2 = [0, 0, 1; 0, 0, 1];
%X = [Xs; Xt];
Y = [zeros(numPos_1,1); zeros(numNeg_1,1)+1; zeros(numPos_2,1)+2; zeros(numNeg_2,1)+3];
gscatter(XX(:,1), XX(:,2),  Y, [c1; c2], '+.+.');  

xlabel(axlb{1})
ylabel(axlb{2})

title(sprintf('%s %.4f', name, metric))
%title(sprintf('%s, %.2f%%',name, metric*100))

end
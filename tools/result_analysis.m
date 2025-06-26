
function [acc, auc, sen, spe, bac]  = result_analysis(pred, prob, labels)
% pred : predicted labels: 1, 0
% prob: predicted scores
% labels: labels for all the samples

T = labels;

index_T1 = find(T == 1); % get the true  1 in groud-truth
index_T2 = find(T == 0);  % get the true  2 in groud-truth
index_P1 = find(pred == 1); %get predicted  1 
index_P2 = find(pred == 0); %get predicted  2 

TP= length( intersect (index_T1, index_P1) );
FP= length(index_P1) - TP;
TN = length( intersect (index_T2, index_P2) );
FN = length(index_P2) - TN;


%acc = nnz(pred == target(mas.te))/length(pred); % total acc
acc = (TP+TN)/(TP+TN+FP+FN);
sen = TP/(TP+FN);
spe = TN/(TN+FP);
bac = (sen+spe)/2;
%%%%% compute AUC %%%%%%%%%%%%%%%
[~,~,~, auc] = perfcurve(labels, prob, '1');

end

function cc = CorrN(Image, Ref)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if(isequal(size(Image), size(Ref))==false)
    error('two image must have same dimension');
end

N = size(Ref, 3);

cc = 0;

for i = 1: N
    cc_t = corr2(Image(:,:, i), Ref(:,:,i));
    if(isnan(cc_t))
        cc_t = 0;
    end
    cc = cc + cc_t;
end

cc = cc/N;

end


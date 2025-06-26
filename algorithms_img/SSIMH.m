function source_new = SSIMH(source, target, param)



%%%========>  set your parameters here <=======%%%%%%%%%%
param.b = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(isequal(size(source), size(target))==false)
    error('two volume must have same dimension');
end

N = size(target, 3);

source_new = source;

b = param.b;

for i = 1: N
    source_new(:, :, i) = FDA_Slice(source(:, :, i), target(:, :, i), b);
end

end

function src_in_trg = FDA_Slice(source_slice, target_slice, b)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
a_src = dct2(source_slice);

a_trg = dct2(target_slice);

a_src(1: b, 1 : b) = a_trg(1: b, 1 : b);

%get the harmonized image
src_in_trg =  idct2(a_src);

end


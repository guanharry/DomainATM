function src_in_trg = FDA_Slice(source_slice, target_slice, b)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
a_src = dct2(source_slice);

a_trg = dct2(target_slice);

a_src(1: b, 1 : b) = a_trg(1: b, 1 : b);

%get the harmonized image
src_in_trg =  idct2(a_src);

end


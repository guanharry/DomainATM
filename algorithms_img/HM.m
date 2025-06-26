function source_new = HM(source, target, param)

% if(isequal(size(source), size(target))==false)
%     error('two volume must have same dimension');
% end

%N = size(target, 3);

Image_S = imresize3(source, size(target));
            
Image_Sh = imhistmatchn(Image_S, target);

source_new = Image_Sh;

end


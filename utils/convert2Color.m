function im=convert2Color(im)
% convert gray image to colr rgb
[~,~,z]=size(im);

if z==1
    
    im=im(:,:,[1 1 1]);
end
end
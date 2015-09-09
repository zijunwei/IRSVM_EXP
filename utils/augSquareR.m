function im=augSquareR(augSz, imsz)
% get sub regions from the image, top, bottom, left, right and center.
% output im cell in xyxy format
if augSz>imsz(1) && augSz>imsz(2)
    
   error('cannot crop image because the cropped region is larger than the boundary \n'); 
end

x=imsz(2);
y=imsz(1);

imcenterx=round(x/2);
imcentery=round(y/2);
half=(augSz/2);



im=cell(5,1);
im{1}=[1,1,augSz,augSz];                                % upper left
im{2}=[1,y-augSz+1,augSz,y];
im{3}=[x-augSz+1,1,x,augSz];
im{4}=[x-augSz+1,y-augSz+1,x,y];
im{5}=round([imcenterx-half+1,imcentery-half+1,imcenterx+half,imcentery+half]);







end
%tstCode2

close all
figure
imshow(im)
 hold on
 re_regions=regionAugment(pboxes(3:4,:),opts);
 drawBoundingBoxes(re_regions, 'xyxy')
close all
figure
imshow(fliplr( im))
 hold on
 
 re_regions=regionAugment(pboxes(3:4,:),opts);
 re_regions=bbFlip(re_regions,size(im,2));
 drawBoundingBoxes(re_regions, 'xyxy')
% this is used to test the dense sampling of vgg neg :
% 
% opts.netname='imagenet-vgg-verydeep-16';
% opts.netdir='./models';
% 
% net=load(fullfile(opts.netdir,opts.netname));

net=vl_simplenn_move(net,'gpu');
img=imread('/nfs/bigbang/zijun/imgnet/imgnet2012/ILSVRC2012_img_val/ILSVRC2012_val_00033334.JPEG');
img=single(img);

img1=gpuArray(single(img));
res1=vl_simplenn(net,img);




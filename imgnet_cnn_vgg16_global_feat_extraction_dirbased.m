
function imgnet_cnn_vgg16_global_feat_extraction_dirbased(input_dir,varargin)
% description:
% cnn file is vgg-verydeep-16 net.
% for each image, augument image by :
% 1. reisze image to be X by 256
% 2. extract 5 regions(4 corners and 1 center)
% 3. flip the image and get same region.
% 
% for each image, use l2 normalized image features and average them to get
% the global representation
% 
% propose N regions, augment them and use them as local features.
% concatnate them together.

imgnetOpts=imgnet_gl_initialization();



outputdir=fullfile(imgnetOpts.glfeat_dir,input_dir);
if ~exist(outputdir,'dir')
    mkdir(outputdir)
end

outputdir=fullfile(outputdir,'%s.mat');
absinputdir=fullfile(imgnetOpts.root,input_dir);

imgs=dir(fullfile(absinputdir,'*.JPEG'));
imgnames={imgs.name};

net=load(('models/imagenet-vgg-verydeep-16.mat'));
net = vl_simplenn_move(net, 'gpu') ;


count=0;
total_time = 0;



args.startid=1;
args.endid=length(imgnames);
args.norm=false;
args=vl_argparse(args,varargin);

opts.topN=100;
opts.sz=256;
opts.step=5;
opts.augN=5;
opts.norm=args.norm;
opts.averageImageV=net.normalization.averageImage(1,1,:);







for i=args.startid:1:args.endid
    fprintf(' processing image : %s  %d \\ [%d - %d] \n', imgnames{i},i,args.startid,args.endid);
    
    im=imread(fullfile(absinputdir,imgnames{i}));
        
    
     [~,~,z]=size(im);
    
    if z==1
        
       fprintf('%s is greylevel img\n',imgnames{i});
       im=im(:,:,[1 1 1]);
    end
    
    count = count + 1;
    tot_th = tic;
    th = tic;
    

    

    % global region feat
    gl_feat=featAugment(im,net,opts);


    fprintf(' [extracting features: %.3fs]\n', toc(th));
    
    
    th=tic;
    save(sprintf(outputdir,imgnames{i}(1:end-5)),'gl_feat','-v7.3')
    
    fprintf(' [saving features: %.3fs]\n', toc(th));
    total_time = total_time + toc(tot_th);
    fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
        total_time/count, total_time);
    
end


        
end


function D = l2norm(D)
            dnorm = sqrt(sum(D.^2,1));
            D = D./repmat(dnorm, size(D,1), 1);
end


function im= imgResz(input_im,scale)
    img_height=size(input_im,1);
    img_width =size(input_im,2);

    
    if img_height>img_width
        
        img_width=scale;
        img_height=NaN;
    else
        
        img_width=NaN;
        img_height=scale;
    end
    
    im=imresize(input_im,[img_height,img_width],'bilinear','antialiasing',false);

end
% change this later
function sub_imgs=augImg(im)

% [im_center_y,im_center_x,~]=   size(im);
im_center_y=ceil(size(im,1)/2);
im_center_x=ceil(size(im,2)/2);

im1=im(1:224,1:224,:);          % leftop
im2=im(end-223:end,1:224,:);    % leftbotton
im3=im(1:224,end-223:end,:);    % righttop
im4=im(end-223:end,end-223:end,:);    % rightbottom
im5=im(im_center_y-112:im_center_y+112-1,im_center_x-112:im_center_x+112-1,:); % center

im6=fliplr(im1);
im7=fliplr(im2);
im8=fliplr(im3);
im9=fliplr(im4);
im10=fliplr(im5);

sub_imgs={im1,im2,im3,im4,im5,im6,im7,im8,im9,im10};

end

function aug_feat=featAugment(im,net,opts)
    im_gl=imgResz(im,opts.sz);
    aug_imgs=augImg(im_gl);
    feat_cell=cell(1,length(aug_imgs));
    for ii=1:length(aug_imgs)
        tmp_im=single(aug_imgs{ii})-net.normalization.averageImage;
        tmp_im=gpuArray(tmp_im);
        res=vl_simplenn(net, tmp_im);
        feat_cell{ii}=res(end-2).x(:);
    end
    feat=cat(2,feat_cell{:});
    if opts.norm
    feat=l2norm(feat);
    end
    aug_feat=l2norm (gather( double(mean(feat,2))));
    
end
function imgnetOpts=imgnet_gl_initialization()
imgnetOpts.root='/nfs/bigbang/zijun/imgnet/imgnet2012';

imgnetOpts.train_dir=[imgnetOpts.root,'/ILSVRC2012_img_train'];
imgnetOpts.val_dir=[imgnetOpts.root,'/ILSVRC2012_img_val'];
imgnetOpts.glfeat_dir=[imgnetOpts.root,'/ILSVRC2012_glfeat_cnn'];
train_dirs=dir(fullfile(imgnetOpts.train_dir,'n*'));
imgnetOpts.wnids={train_dirs.name};

end
% use the whole image resized to 224 by 224 for image classification
%  the verification of whole algorithm

%%%%%%%%%%%%%%%%%%%%%%%%%% Remember to run rcnnDarwin_prerun to setup %%%%
function voc07_spp_vgg16_feat_extraction(setname,varargin)
% prerun;

VOCinit();
gtids=textread(sprintf(VOCopts.imgsetpath,setname),'%s');
args.startid=1;
args.endid=length(gtids);
args.norm=true;
args=vl_argparse(args,varargin);

if args.norm==true
featdir=[VOCopts.featpath,'spp_vgg16_featbags_gpu'];
else
featdir=[VOCopts.featpath,'spp_vgg16_featbags_gpu_nn'];
end
if ~exist(featdir,'dir')
   mkdir(featdir); 
end
featdir=fullfile(featdir,'%s.mat');


net=load(('models/imagenet-vgg-verydeep-16.mat'));
net.layers=net.layers( 1:(length(net.layers)-7) );
net = vl_simplenn_move(net, 'gpu') ;

spp_pooler=load('spp_pooler.mat');
spp_pooler=spp_pooler.spp_pooler;
net = vl_simplenn_move(net, 'gpu') ;

opts.topN=100;
opts.ratio=224/256;
opts.sz=448;
opts.step=5;
opts.augN=5;
opts.averageImageV=net.normalization.averageImage(1,1,:);


count=0;
total_time = 0;




for i=args.startid:args.endid
    fprintf(' processing image : %s  %d \\ [%d - %d] \n', gtids{i},i,args.startid,args.endid);
    
    im=imread(sprintf(VOCopts.imgpath,gtids{i}));

    
    count = count + 1;
    tot_th = tic;
    th = tic;
    

    

    % global region feat
    gl_regions=regionAugment([1,1,size(im,2),size(im,1)],opts);
    
    % img region proposl
    bbs=[];
    load(sprintf(VOCopts.regproposalpath,gtids{i}));
    if isempty(bbs)
       error('%s doesn''t have corresponding propsoal regios\n ') 
    end
    
    [~,o]=sort(bbs(:,end),'descend');
    pboxes=bbs(o,1:4);
    
    if length(pboxes)>opts.topN;
       pboxes=pboxes(1:opts.topN,:); 
    end
    
    re_regions=regionAugment(pboxes,opts);
    
    im_r=fliplr(im);
    re_regions_r=bbFlip(re_regions,size(im,2));
    feats_f=spp_feat_s( im ,[gl_regions;re_regions],  opts.sz,opts.averageImageV,net,spp_pooler);
    feats_r=spp_feat_s(im_r,[gl_regions;re_regions_r],opts.sz,opts.averageImageV,net,spp_pooler);
    
    assert(size(feats_f,1)==size(feats_r,1));
    gl_feat=[feats_f(:,1:5),feats_r(:,1:5)];
    if args.norm
    gl_feat=l2norm(gl_feat);
    end
    gl_feat= ( double(mean(gl_feat,2)));
    gl_feat=l2norm(gl_feat);
    
    
    
     re_feat_cell=cell(1,size(pboxes,1));
     for ii=1:1:size(pboxes,1)
        idx=(opts.augN+ii):size(pboxes,1):size(feats_f,2);
        re_feat_s=[feats_f(:,idx),feats_r(:,idx)];
        if args.norm
         re_feat_s=l2norm(re_feat_s);
        end
       re_feat_s= ( double(mean(re_feat_s,2)));
       re_feat_cell{ii}=re_feat_s;
     end
      re_feat=cat(2,re_feat_cell{:});
      re_feat=l2norm(re_feat); 

    fprintf(' [extracting features: %.3fs]\n', toc(th));
    
    
    th=tic;
    save(sprintf(featdir,gtids{i}),'gl_feat','re_feat','-v7.3')
    
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


function regions=regionAugment(bbs,opts)
ratio=opts.ratio;
im_h=bbs(:,4)-bbs(:,2);
im_w=bbs(:,3)-bbs(:,1);
r_im_h=floor(im_h.*ratio);
r_im_w=floor( im_w.*ratio);

imcenter_x=floor(im_w/2)+bbs(:,1)-1;
imcenter_y=floor(im_h/2)+bbs(:,2)-1;
half_imrh=floor(r_im_h/2);
half_imrw=floor(r_im_w/2);
aug_regions=cell(1,5);
aug_regions{1}=  [bbs(:,1)                      ,bbs(:,2),                        bbs(:,1)+r_im_w-1,                    bbs(:,2)+r_im_h-1];
aug_regions{2}=  [bbs(:,1)                      ,bbs(:,2)+im_h-r_im_h,            bbs(:,1)+r_im_w-1,                    bbs(:,2)+im_h-1];
aug_regions{3}=  [bbs(:,1)+im_w-r_im_w          ,bbs(:,2),                        bbs(:,1)+im_w-1,                      bbs(:,2)+r_im_h-1];
aug_regions{4}=  [bbs(:,1)+im_w-r_im_w          ,bbs(:,2)+im_h-r_im_h+1,          bbs(:,1)+im_w-1,                      bbs(:,2)+im_h-1];
aug_regions{5}=  [imcenter_x-half_imrw          ,imcenter_y-half_imrh,            imcenter_x+half_imrw-1,               imcenter_y+half_imrh-1];

regions=cat(1,aug_regions{:});

end
function bb_flipped=bbFlip(bbs,im_w)
bb_flipped= [ im_w- bbs(:,3),bbs(:,2),im_w-bbs(:,1),bbs(:,4)];
end

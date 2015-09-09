function glfeats=extract_feat(input_img,opts)
% extract_global features
% the input_img is alread





if strcmpi(opts.augmenttype,'default_aug')
    input_img=imScale(input_img,opts.imsz(1));
    imsize=size(input_img);
    
    augBB=augSquareR(opts.patchsz,imsize);
    
    im_f=input_img;
    im_r=fliplr(input_img);
    glfeats=cnnfeats(im_f,im_r,augBB,opts.net);
elseif strcmpi(opts.augmenttype,'none_aug')
    input_img=imScale(input_img,opts.imsz(1));
    imsize=size(input_img);
    
    augBB=augSquareR(opts.patchsz,imsize);
    
    im_f=input_img;
    im_r=fliplr(input_img);
    glfeats=cnnfeats_im(im_f,augBB(end),opts.net);
elseif strcmpi(opts.augmenttype,'densesampling')
    im_f=input_img;
    im_r=fliplr(im_f);
    glfeats{1}=cnnfeatsdensesampling(im_f,opts);
    glfeats{2}=cnnfeatsdensesampling(im_r,opts);
    
end
glfeats=cat(2,glfeats{:});
glfeats=gather(glfeats);



if opts.norm==true
    glfeats=l2norm(glfeats);
    
end

glfeats=sum(glfeats,2);
glfeats=l2norm(glfeats);

end

function glfeats=cnnfeatsdensesampling(im,opts)
glfeats=0;
for i=1:1:length(opts.imsz)
    imi=imScale(im,opts.imsz(i));
    imi=single(imi);
    sz=size(imi);
    imi=imi- repmat(opts. averageImageV,sz(1),sz(2));
    imi=gpuArray( (imi));
    res=vl_simplenn(opts.net,imi);
    x=res(end).x;
    x=sum(x,1);
    x=sum(x,2);
    x=x(:);
end
glfeats=glfeats+x;

end
function glfeats=cnnfeats(im_f,im_r,augBB,net)
glfeatsf=cnnfeats_im(im_f,augBB,net);
glfeatsr=cnnfeats_im(im_r,augBB,net);
glfeats={glfeatsf{:};glfeatsr{:}};
end
function glfeats=cnnfeats_im(im,augBB,net)
glfeats=cell(length(augBB),1);
for i=1:1:length(augBB)
    bb=augBB{i};
    tmpim=im(bb(2):bb(4),bb(1):bb(3),:);
    tmpim=single(tmpim)-net.normalization.averageImage;
    tmpim=gpuArray(tmpim);
    res=vl_simplenn(net,tmpim);
    glfeats{i}=res(end).x(:);
    % -------------------------------------
end

end


function glfeats=sppfeats(im_f,im_r,augBB,opts)
glfeatsf=sppfeats_im(im_f,augBB,opts);
glfeatsr=sppfeats_im(im_r,augBB,opts);
glfeats=[glfeatsf,glfeatsr];
end


function glfeats=sppfeats_im(im,augBB,opts)
regions=cat(1,augBB{:});
glfeats=spp_feat_s( im ,regions,  opts.imsz,opts.averageImageV,opts.net,opts.spp_pooler);

end




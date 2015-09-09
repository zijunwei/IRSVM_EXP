function glfeats=extract_feat(input_img,opts)
% extract_global features
% the input_img is alread

input_img=imScale(input_img,opts.imsz);
imsize=size(input_img);

augBB=augSquareR(opts.patchsz,imsize);

im_f=input_img;
im_r=fliplr(input_img);



if  strcmpi( opts.method,'cnn')
    if strcmpi(opts.augmenttype,'default_aug')
    glfeats=cnnfeats(im_f,im_r,augBB,opts.net);
    elseif strcmpi(opts.augmenttype,'none_aug')
    glfeats=cnnfeats_im(im_f,augBB(end),opts.net);
    end
    glfeats=cat(2,glfeats{:});
    glfeats=gather(glfeats);
    
%    uses the center image patch as the target 
        
        
    
    
    
elseif strcmpi(opts.method,'spp')
    
    glfeats=sppfeats(im_f,im_r,augBB,opts);
    
end


if opts.norm==true
    glfeats=l2norm(glfeats);
    
end

glfeats=sum(glfeats,2);
glfeats=l2norm(glfeats);

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




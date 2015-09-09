function [ feat,boxes_scales ] = spp_feat_s_bkup( im ,boxes,spm_im_size,averageImageV,net,spp_pooler)
%FEAT_EXTRACTION Summary of this function goes here
%   Detailed explanation goes here

% extract features:
im_width = size(im, 2);
im_height = size(im, 1);
feat.im_height = im_height;
feat.im_width = im_width;
feat.scale = spm_im_size;
feat.rsp = {};

    scale = spm_im_size;
    

    
    % resize min(width, height) to scale
    if (im_width < im_height)
        im_resized_width = scale;
        im_resized_height = im_resized_width * im_height / im_width;
    else
        im_resized_height = scale;
        im_resized_width = im_resized_height * im_width / im_height;
    end

    % We turn off antialiasing to better match OpenCV's bilinear 
    resized_im = imresize(im, [im_resized_height, im_resized_width], 'bilinear', 'antialiasing', false);


      % res is height,width,channel
      
      input_im=single(resized_im);
      sz=size(input_im);
      input_im=input_im- repmat( averageImageV,sz(1),sz(2));
      input_im=gpuArray(input_im);
       res=vl_simplenn(net, input_im);
      feat.rsp{1} = gather( res(end).x);
      clearvars input_im 

% feature pooling:
min_img_sz = min(feat.im_height, feat.im_width);
% feat.scale = feat.scale(:)'; 
expected_scale = spp_pooler.expected_scale(min_img_sz, boxes, spp_pooler);
best_scale_ids=ones(size(boxes,1),1);
% [~, best_scale_ids] = min(abs(bsxfun(@minus, feat.scale, expected_scale(:))), [], 2);
boxes_scales = feat.scale(best_scale_ids(:));
scaled_boxes = bsxfun(@times, (boxes - 1), (boxes_scales(:) - 1)) / (min_img_sz - 1) + 1;
% rep_boxes = spp_pooler.response_boxes(scaled_boxes, spp_pooler);
%     rsp_keys = [rep_boxes, best_scale_ids];
%     [~, ia] = unique(rsp_keys, 'rows');
feat = spm_pool(feat.rsp, spp_pooler.spm_divs, scaled_boxes', best_scale_ids, ...
    spp_pooler.offset0, spp_pooler.offset, spp_pooler.step_standard);
end


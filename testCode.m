%tst:
% run matlab/vl_setupnn
% 
% % download a pre-trained CNN from the web
% 
% net = load('models/imagenet-vgg-verydeep-16.mat') ;
% 
% % obtain and preprocess an image
% im = imread('pepper.jpg') ;
% im_ = single(im) ; % note: 255 range
% 
% im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
% im_ = im_ - net.normalization.averageImage ;
% 
% % run the CNN
% 
% im_=imresize(im_,2);
% net = vl_simplenn_move(net, 'gpu') ;
% res = vl_simplenn(net, gpuArray( im_)) ;
% 
% % show the classification result
% % scores = squeeze(gather(res(end).x)) ;
% % [bestScore, best] = max(scores) ;
% % figure(1) ; clf ; imagesc(im) ;
% % title(sprintf('%s (%d), score %.3f',...
% % net.classes.description{best}, best, bestScore)) ;
clear
cuda7_0path='/usr/local/cuda';
addpath matlab
vl_compilenn('enableGpu', true, 'cudaRoot', cuda7_0path);
run matlab/vl_setupnn
% addpath('matlab')
vl_test_nnlayers(true);

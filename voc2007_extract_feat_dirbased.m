function voc2007_extract_feat_dirbased(varargin)
% For developper Zijun:
% we will derive imagenet dataset from here, something we need to pay
% attentiopn to is :
% 1. we need to change the way we acquire imglist.
% 2. we need to change the dir we save features

% before running make sure you run prerun...


% for users:
% please read the params set by opts.
% without specification, the spp/cnn data will be save in:
% <your voc
% dir>/VOC2007/ExtractedFeatures/method_featturetype_networkname_augmentationtype
opts.imgset='trainval';  % will be trianval and test
opts.netname='imagenet-vgg-verydeep-16';
opts.method='cnn';           % cnn spp
opts.feattype='local';      % global, local
opts.augmenttype='densesampling';  %possible interfaces for further image augment type default_aug and none_aug and densesampling
opts.startid=1;
opts.endid=[];
opts.gpu=1;
opts.task='det';

opts.poolername='./spp_pooler.mat';
opts.netdir='./models';
opts.impps=[];     % dir to save image proposals
%-------------- nn related params
opts.imsz=256;      %scale your image to this
opts.patchsz=224;   %sample sub patches from this
opts.norm=false;
opts.topN=2000;

opts=vl_argparse(opts,varargin);

addpath('./utils');

gpuDevice(opts.gpu);

VOCinit;
opts.dataset=VOCopts;
opts.impps=fullfile(opts.dataset.datadir,opts.dataset.dataset,'RegionProposal');
%-----------------target disk full ,change to a temparay place -------
% savedir=fullfile( opts.dataset.datadir,opts.dataset.dataset,'ExtractedFeatures');
savedir='/nfs/bigbang/zijun/voc2007_tmp_savesapce';
%---------------------------------------------------------------
savedir=fullfile(savedir,sprintf('%s_%s_%s_%s_%s',opts.task,opts.method,opts.feattype,opts.netname,opts.augmenttype));
if ~exist(savedir,'dir')
    mkdir(savedir);
end




% load models
opts.net=load(fullfile(opts.netdir, opts.netname));
opts.net=vl_simplenn_move(opts.net,'gpu');
if strcmpi(opts.method,'cnn')
    opts.spp_pooler=[];
    % ---- chose end-2  as the output layer
    
    opts.net.layers=opts.net.layers(1:end-2);
    
elseif strcmpi(opts.method,'spp')
    opts.spp_pooler=load(fullfile( opts.netdir, opts.poolername));
    opts.spp_pooler=opts.spp_pooler.spp_pooler;
    opts.averageImageV=opts.net.normalization.averageImage(1,1,:);
    % ---- chose end-7  as the output layer
    
    opts.net.layers=opts.net.layers(1:end-7);
end


gtids=textread(sprintf(opts.dataset.imgsetpath,opts.imgset),'%s');
imagelist=cellfun(@(x) sprintf(opts.dataset.imgpath,x),gtids,'UniformOutput', false);

if isempty(opts.endid)
    opts.endid=length(imagelist);
end

imagelist=imagelist(opts.startid:opts.endid);

% no matter local or global, come in this side
extract_feat_imglist(imagelist,savedir,opts);


end
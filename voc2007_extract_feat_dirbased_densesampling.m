function voc2007_extract_feat_dirbased_densesampling(varargin)
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
opts.feattype='global';      % global, local
opts.augmenttype='densesampling';  % image augment type: 1 default_aug, 2 none_aug 3. densesampling
opts.startid=1;
opts.endid=[];
opts.gpu=1;
opts.task='cls'; % cls for classification , det for detection
opts.netdir='./models';
%-------------- nn related params
opts.imsz=[  256,384,512 ];      %scale your image to this
opts.norm=false;
opts.topN=2000;

opts=vl_argparse(opts,varargin);


gpuDevice(opts.gpu);

VOCinit;
opts.dataset=VOCopts;
opts.impps=fullfile(opts.dataset.datadir,opts.dataset.dataset,'RegionProposal');
savedir=fullfile( opts.dataset.datadir,opts.dataset.dataset,'ExtractedFeatures');
opts.savedir=fullfile(savedir,sprintf('%s_%s_%s_%s_%s',opts.task,opts.method,opts.feattype,opts.netname,opts.augmenttype));
if ~exist(opts.savedir,'dir')
    mkdir(opts.savedir);
end




% load models
opts.net=load(fullfile(opts.netdir, opts.netname));
opts.net=vl_simplenn_move(opts.net,'gpu');

% ---- chose end-2  as the output layer
opts.net.layers=opts.net.layers(1:end-2);
opts.averageImageV=opts.net.normalization.averageImage(1,1,:);




gtids=textread(sprintf(opts.dataset.imgsetpath,opts.imgset),'%s');
imagelist=cellfun(@(x) sprintf(opts.dataset.imgpath,x),gtids,'UniformOutput', false);

if isempty(opts.endid)
    opts.endid=length(imagelist);
end

imagelist=imagelist(opts.startid:opts.endid);

% no matter local or global, come in this side
extract_feat_imglist(imagelist,opts.savedir,opts);


end
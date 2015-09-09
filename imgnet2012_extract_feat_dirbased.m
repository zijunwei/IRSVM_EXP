function imgnet2012_extract_feat_dirbased(varargin)
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
opts.imgset='train'; % will be train1 (1-500) train2 to split. ofcourse you can do more split
opts.netdir='./models';
opts.netname='imagenet-vgg-verydeep-16';
opts.poolername='./spp_pooler.mat';
opts.method='cnn';           % cnn spp
opts.feattype='local';      % global, local
opts.augmenttype='none_aug';  %possible interfaces for further image augment type
opts.startid=1;
opts.endid=[];
opts.gpu=1;
opts.topN=100;
%-------------- nn related params
opts.imsz=256;      %scale your image to this
opts.patchsz=224;   %sample sub patches from this
opts.norm=false;

opts=vl_argparse(opts,varargin);

addpath('./utils');

gpuDevice(opts.gpu);

IMGNETinit;
opts.dataset=IMGNETopts;
savedir=fullfile( opts.dataset.root,'ExtractedFeatures');
savedir=fullfile(savedir,sprintf('%s_%s_%s_%s',opts.method,opts.feattype,opts.netname,opts.augmenttype));
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

  impps=fullfile(IMGNETopts.root,sprintf('ILSVRC2012_rp_%s',opts.imgset));

switch opts.imgset
    case 'train'
        if isempty(opts.endid)
            opts.endid=opts.dataset.nclasses;
        end
        startid=opts.startid;
        endid=opts.endid;
        
        for i=startid:endid
            fprintf('processing %s , totoal progress [ %d / %d ]\n',IMGNETopts.wnids{i},i,endid-startid);
            sub_save_dir=fullfile(savedir,IMGNETopts.trainset,IMGNETopts.wnids{i});
            if ~exist(sub_save_dir,'dir')
                mkdir(sub_save_dir);
            end
            % imglist:
            gtids=dir(fullfile(IMGNETopts.root,IMGNETopts.trainset,IMGNETopts.wnids{i},'*.JPEG'));
            gtids={gtids.name};
            imagelist=cellfun(@(x)fullfile(IMGNETopts.root,IMGNETopts.trainset,IMGNETopts.wnids{i},x),gtids,'UniformOutput', false);
            opts.impps=fullfile( impps,IMGNETopts.wnids{i});
            
            opts.startid=1;
            opts.endid=length(imagelist);
%             if strcmpi(opts.feattype,'global')
                extract_feat_imglist(imagelist,sub_save_dir,opts);
%             else
%                 
%                 
%             end
            
        end
        
    case 'val'
        sub_save_dir=fullfile(savedir,IMGNETopts.valset);
        if ~exist(sub_save_dir,'dir')
            mkdir(sub_save_dir);
        end
        % imglist:
        gtids=dir(fullfile(IMGNETopts.root,IMGNETopts.valset,'*.JPEG'));
        gtids={gtids.name};
        imagelist=cellfun(@(x)fullfile(IMGNETopts.root,IMGNETopts.valset,x),gtids,'UniformOutput', false);
        opts.startid=1;
        opts.endid=length(imagelist);
        opts.impps=impps;
        extract_feat_imglist(imagelist,sub_save_dir,opts);
      
    case 'test'
        sub_save_dir=fullfile(savedir,IMGNETopts.testset);
        if ~exist(sub_save_dir,'dir')
            mkdir(sub_save_dir);
        end
        % imglist:
        gtids=dir(fullfile(IMGNETopts.root,IMGNETopts.testset,'*.JPEG'));
        gtids={gtids.name};
        imagelist=cellfun(@(x)fullfile(IMGNETopts.root,IMGNETopts.testset,x),gtids,'UniformOutput', false);
        opts.startid=1;
        opts.endid=length(imagelist);
        extract_feat_imglist(imagelist,sub_save_dir,opts);
       
        
        
end

end





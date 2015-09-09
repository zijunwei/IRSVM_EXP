% the baseline:
% data collection information:
% 1.method: cnn
% 2.glboal feature with 10 augmentation and average (without normalization)
% 3.least square svm
% 4. trained using vgg 16 deep
function voc2007_baseline
VOCinit;

% -------------------these configs helps with the name of the directory for saving the
% features
opts.netname='imagenet-vgg-verydeep-16';
opts.method='cnn';                % cnn spp


opts.feattype='global+avglocal';           % global, local
opts.augmenttype1='default_aug';
opts.augmenttype2='none_aug';%possible interfaces for further image augment type
dirname=sprintf('%s_%s_%s_%s_%s',opts.method,opts.feattype,opts.netname,opts.augmenttype1,opts.augmenttype2);

% -------------------save cache in order to avoid re-reading
cache_dir=['./voc2007_cache/',dirname];

if ~exist(cache_dir,'dir')
    mkdir(cache_dir);
end

data_cahce='data.mat';
result_cache='result.mat';

if exist(fullfile(cache_dir,data_cahce),'file')
    fprintf('feature cache exists, reading directly from cache dir %s \n',cache_dir);
    load(fullfile(cache_dir,data_cahce));
    
else
    
    fprintf('feature cache NOT exists, reading directly from single files %s \n',dirname);
    
    % ----------    feature extraction -------------------------------------
    % get lables:
    VOCopts.lbset=[VOCopts.datadir,VOCopts.dataset,'/ImageSets/Main/%s.mat'];
    trSum=load(sprintf( VOCopts.lbset,'trainval'));
    tstSum=load(sprintf(VOCopts.lbset,'test'));
    trLb=trSum.lbs;
    tstLb=tstSum.lbs;
    
    trD_global=[];
    tstD_global=[];
    trD_local=[];
    tstD_local=[];
    %features loading
    if ~isempty( strfind(   opts.feattype,'global'))
        dirname1=sprintf('%s_%s_%s_%s',opts.method,'global',opts.netname,opts.augmenttype1);
        
        featdir=fullfile(VOCopts.datadir,VOCopts.dataset,'ExtractedFeatures',dirname1);
        
        % tr features
        trCell=readFeatfromFile(trSum,featdir,'setname','trainval_global');
        trD_global=cat(2,trCell{:});
        
        % tst featrues:
        tstCell=readFeatfromFile(tstSum,featdir,'setname','test_global');
        tstD_global=cat(2,tstCell{:});
        %         save(fullfile(cache_dir,data_cahce),'trD','tstD','trLb','tstLb','-v7.3');
    end
    if ~isempty(strfind(opts.feattype,'local'))
        dirname2=sprintf('%s_%s_%s_%s',opts.method,'local',opts.netname,opts.augmenttype2);
        tmpsavedir='/nfs/bigbang/zijun/voc2007_tmp_savesapce';
        featdir=fullfile(tmpsavedir,dirname2);
        trCell=readFeatfromFile(trSum,featdir,'setname','trainval_local','avg',true);
        trD_local=cat(2,trCell{:});
        
        % tst featrues:
        tstCell=readFeatfromFile(tstSum,featdir,'setname','test_local','avg',true);
        tstD_local=cat(2,tstCell{:});
    end
    trD=[trD_global;trD_local];
    tstD=[tstD_global;tstD_local];
    
    trD=l2norm(trD);
    tstD=l2norm(tstD);
    save(fullfile(cache_dir,data_cahce),'trD','tstD','trLb','tstLb','-v7.3');
    
    
end
% ------------- run svm classifier to show meanAP


lambda_factor=1e-4;
aps = zeros(length(VOCopts.classes), 1);

for kk=1:length(VOCopts.classes)
    fprintf('Running for class %s\n', VOCopts.classes{kk});
    
    
    trLb_i  = trLb(:,kk);
    trIdxs = trLb_i ~= 0;
    trLb_i   = trLb_i(trIdxs);
    
    tstLb_i  = tstLb(:,kk);
    tstIdxs = tstLb_i ~= 0;
    tstLb_i = tstLb_i(tstIdxs);
    
    lambda = lambda_factor*length(trLb_i);
    [w,b] = ML_Ridge.ridgeReg(trD(:,trIdxs), trLb_i, lambda, ones(size(trLb_i)));
    
    initScore = tstD(:,tstIdxs)'*w + b;
    aps(kk,1) = ml_ap(initScore, tstLb_i, 0);
end



%print & save results out:
for kk=1:1:length(VOCopts.classes)
    fprintf('%s : \t %2f \n',VOCopts.classes{kk},aps(kk));
end
fprintf('mean Average Precison is %2f \n',mean(aps));
save(fullfile(cache_dir,result_cache),'aps');
end






function  trCell=readFeatfromFile(trSum,featdir,varargin)

opts.avg=false;
opts.setname='';
opts=vl_argparse(opts,varargin);
printinfo=sprintf('reading %s data',opts.setname);

numimg=length(trSum.imIds);
% for debug use
% numimg=5;
trCell=cell(1,numimg);
for i=1:1:numimg
    ml_progressBar(i,numimg,printinfo);
    sgFeat=load(fullfile(featdir,sprintf('%.06d.mat' ,trSum.imIds(i))));
    if isfield(sgFeat,'gl_feat')
        trCell{i}=sgFeat.gl_feat;
    elseif isfield(sgFeat,'re_feat')
        trCell{i}=sgFeat.re_feat;
    end
    
    if opts.avg
        
        trCell{i}= l2norm(mean(trCell{i},2)) ;
    end
end


end
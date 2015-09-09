% the baseline:
% data collection information:
% 1.method: cnn
% 2.glboal feature with 10 augmentation and average (without normalization)
% 3.least square svm
% 4. trained using vgg 16 deep
function voc2007_irsvm
VOCinit;

% -------------------these configs helps with the name of the directory for saving the
% features
opts.netname='imagenet-vgg-verydeep-16';
opts.method='cnn';                % cnn spp cnn irsvm
opts.feattype='irsvm';

opts.augmenttype_global='default_aug';
opts.augmenttype_local='default_aug';%possible interfaces for further image augment type
dirname=sprintf('%s_%s_%s_%s_%s',opts.method,opts.feattype,opts.netname,opts.augmenttype_global,opts.augmenttype_local);

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
    
    
    % global features loading
    dirname1=sprintf('%s_%s_%s_%s',opts.method,'global',opts.netname,opts.augmenttype_global);
    
    featdir_global=fullfile(VOCopts.datadir,VOCopts.dataset,'ExtractedFeatures',dirname1);
    
%     % ---tr global features
%     trCell=readFeatfromFile(trSum,featdir,'setname','trainval_global');
%     %     trD_global=cat(2,trCell{:})
%     
%     % ---tst global featrues:
%     tstCell=readFeatfromFile(tstSum,featdir,'setname','test_global');
    %     tstD_global=cat(2,tstCell{:});
    
    dirname2=sprintf('%s_%s_%s_%s',opts.method,'local',opts.netname,opts.augmenttype_local);
    tmpsavedir='/nfs/bigbang/zijun/voc2007_tmp_savesapce';
    featdir_local=fullfile(tmpsavedir,dirname2);
    [trD,trBags]=readFeatBagsfromFile(trSum, featdir_global,featdir_local,'setname','trainval');
    [tstD,tstBags]=readFeatBagsfromFile(tstSum,featdir_global,featdir_local,'setname','test');
    
    
    
%     % ---tr local features
%     trLocalCell=readFeatfromFile(trSum,featdir,'setname','trainval_local');
%     % ---tst local featrues:
%     tstLocalCell=readFeatfromFile(tstSum,featdir,'setname','test_local');
%     %     tstD_local=cat(2,tstCell{:});
%     
%     %     trD=[trD_global;trD_local];
%     %     tstD=[tstD_global;tstD_local];
%     %
%     %     trD=l2norm(trD);
%     %     tstD=l2norm(tstD);
%     
%     % save dosen't help with memory save
%     %     save(fullfile(cache_dir,data_cahce),'trCell','tstCell','trLocalCell','tstLocalCell','trLb','tstLb','-v7.3');
%     
%     
%     % ------- assemble feats
%     % training feat:
%     featdim=size(trCell{1},1);
%     
%     
%     
%     trN=length(trCell);
%     trD=zeros(featdim*2,trN);
%     trBags=cell(trN,1);
%     for ii=1:1:trN
%         numlocal=size(trLocalCell{ii},2);
%         avglocal=l2norm( mean(trLocalCell{ii},2));
%        trD(:,ii)=  l2norm(  [trCell{ii};avglocal]);
%        trBags{ii}=  l2norm( [repmat(trCell{ii},1,numlocal);trLocalCell{ii}]);
%        
%     end
%     
%     
%     % testing feat
%     tstN=length(tstCell);
%     tstD=zeros(featdim*2,tstN);
%     tstBags=cell(tstN,1);
%     for ii=1:1:tstN
%         numlocal=size(tstLocalCell{ii},2);
%         avglocal=l2norm( mean(tstLocalCell{ii},2));
%         tstD(:,ii)=  l2norm(  [tstCell{ii};avglocal]);
%         tstBags{ii}=  l2norm( [repmat(tstCell{ii},1,numlocal),tstLocalCell{ii}]);
%        
%     end
%     
end
% ------------- run svm classifier to show meanAP


% clearvars tstCell trCell tstLocalCell trLocalCell

lambda_factor=1e-4;
aps = zeros(length(VOCopts.classes), 2);

% compute trD, tstD, ... and so on...
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
    fprintf('baseline score is %5.2f \n',100*aps(kk,1))
    
    irsvmopts.initOpt = 'wb';
    irsvmopts.w = w;
    irsvmopts.b = b;
    irsvmopts.nIter = 100;
    irsvmopts.nThread = 1;
    
    irsvmopts.methodName = 'IRLSSVM6';
    irsvmopts.initOpt = 'mean';
    
    fprintf('running for irsvm...\n');
    [w, b, s, objVals] = MIR_IRLSSVM6.train(trBags(trIdxs), trLb_i, lambda, irsvmopts);
    tstScore = M_IRSVM.predict(tstBags(tstIdxs), w, b, s);
    aps(kk,2) = ml_ap(tstScore, tstLb_i, 0);
    classap=aps(kk,:);
    fprintf('%s, start: %5.2f, IRLSSVM6: %5.2f\n', ...
        VOCopts. classes{kk}, 100*classap);
    fprintf('#non-zero s: %d\n\n\n', sum(s > 1e-3));
    
    
    save( fullfile(cache_dir,sprintf('irsvmparams_%s.mat',VOCopts.classes{kk})),'w','b');
    
end


save(fullfile(cache_dir,result_cache),'aps');
%print & save results out:
for kk=1:1:length(VOCopts.classes)
    fprintf('%s : \t %5.4f  \t %5.4f \n',VOCopts.classes{kk},aps(kk,1)*100,aps(kk,2)*100);
end
fprintf('mean Average Precison of baseline is %5.4f, irsvm results is %5.4f \n',mean(aps(:,1)),mean(aps(:,2)));
save(fullfile(cache_dir,result_cache),'aps');
end



function combineFeatBag=combineFeat(glfeatCell,refeatCell)
assert(length(glfeatCell)==length(refeatCell));
k=length(glfeatCell);
combineFeat=cell(k,1);
for i=1:1:k
    m=size(refeatCell{i},2);
    combineFeat{i}=[repmat(glfeatCell{i},1,m);refeatCell{i}];
    
    
end
% combineFeatBag=cellfun(@(x)[repmat(glfeatCell)])


end



function [trD,trBags]=readFeatBagsfromFile(trSum, featdir_global,featdir_local,varargin)
opts.setname='';
opts=vl_argparse(opts,varargin);
printinfo=sprintf('reading %s data',opts.setname);

numimg=length(trSum.imIds);
% for debug use
% numimg=5;


trBags=cell(1,numimg);
trCell=cell(1,numimg);
for i=1:1:numimg
    ml_progressBar(i,numimg,printinfo);
    sgFeatg=load(fullfile(featdir_global,sprintf('%.06d.mat' ,trSum.imIds(i))));
    sgFeatl=load(fullfile(featdir_local,sprintf('%.06d.mat' ,trSum.imIds(i))));
       glfeat=sgFeatg.gl_feat;
        refeat=sgFeatl.re_feat;
   trCell{i}= l2norm( [  glfeat;   l2norm( sum(refeat,2) )   ]);
   nlocals=size(refeat,2);
   trBags{i}=l2norm ([repmat(glfeat,1,nlocals);refeat]);
   
end
    
    
trD=cat(2,trCell{:});

end

function  trCell=readFeatfromFile(trSum,featdir,varargin)

opts.avg=false;
opts.setname='';
opts=vl_argparse(opts,varargin);
printinfo=sprintf('reading %s data',opts.setname);

numimg=length(trSum.imIds);
% for debug use

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
        
        trCell{i}=mean(trCell{i},2) ;
    end
end


end
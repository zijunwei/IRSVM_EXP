% add liblinear path:
addpath('liblinear/matlab');

%load data if not exist
if ~exist('multiclslb','var')
    load('imgnetdata.mat');
end


% select subset:

ncls=100;    % select top ncls from all data
nsample=100; % for each class, sample nsample from data.

ptrdata=cell(ncls,1);
pvaldata=cell(ncls,1);
ptrlb=cell(ncls,1);
pvallb=cell(ncls,1);
for i=1:1:ncls
    ml_progressBar(i,ncls);
    allsamples=find(trlb==i);
    ssamples=length(allsamples);
    idxs=randperm(ssamples);
    tridxs=idxs(1:nsample);
    validx=idxs(end-nsample+1:end);
    ptrdata{i}=trd(:,allsamples(tridxs));
    pvaldata{i}=trd(:,allsamples(validx));
    ptrlb{i}=i*ones(1,nsample);
    pvallb{i}=i*ones(1,nsample);
    
end

ptrdata=cat(2,ptrdata{:});
pvaldata=cat(2,pvaldata{:});
ptrlb=cat(2,ptrlb{:});
pvallb=cat(2,pvallb{:});

% start to test different parameters of liblinear:

% 1. using regression : seems not to be correct method.
paramstr1=' -s 2 -c 10 -B -1 ';
ptrsparse=sparse(double( ptrdata));
pvalsparse=sparse(double(pvaldata));
libtimer=tic;
multi_models=liblineartrain(ptrlb',(ptrsparse),paramstr1,'col');
fprintf('done, time use: %5.3f \n',toc(libtimer));
[predict_label, accuracy, dec_values] = liblinearpredict(pvallb', (pvalsparse), multi_models,[],'col');



% %% get top 5 error
cor=0;
multi_lable=multi_models.w*pvaldata;
for i=1:1:size(multi_lable,1)
    ml_progressBar(i,size(multi_lable,1));
    sdet=multi_lable(:,i);
   [~,idx] =sort(sdet);
   top5=idx(1:5);
   if find(top5,pvallb(i))
      cor=cor+1; 
   end
end
fprintf('top5 error is %5.3f\n',1-cor/size(multi_lable,1));
% % % % % % % %2. using each class as a single one vs all ...
% 
% for i=1:1:ncls
%     tmptrlb=ptrlb;
%     tmpvallb=pvallb;
%     trpos=(tmptrlb==i);
%     trneg=(tmptrlb~=i);
%     valpos=(tmpvallb==i);
%     valneg=(tmpvallb~=i);
%     
%     tmptrlb(trpos)=1;
%     tmptrlb(trneg)=-1;
%     tmpvallb(trpos)=1;
%     tmpvallb(trneg)=-1;
%     
%     % train a binayr classifier on each class:
%     multi_models=liblineartrain(tmptrlb',(ptrsparse),' -s 1 ','col');
%     [predict_label, accuracy, dec_values] = liblinearpredict(tmpvallb', (pvalsparse), multi_models,[],'col');
%     
% end
% 
% 
% 
% % 3 test on the whole dataset:
% %%
% cparam=1e-4 * size(trd,2);
% paramstr1=sprintf( ' -s 1 -c %d -B -1 ',cparam);
% trsparse=sparse(double( trd));
% valsparse=sparse(double(vald));
% lltt=tic;
% fprintf('training started \n')
% multi_models=liblineartrain(trlb',(trsparse),paramstr1,'col');
% % llt=toc(lltt);
% fprintf('%5.3f seconds used\n',toc(lltt));
% [predict_label, accuracy, dec_values] = liblinearpredict(vallb', (valsparse), multi_models,[],'col');
% 
% 
% %% get top 5 error
% cor=0;
% multi_lable=multi_models.w*vald;
% for i=1:1:size(multi_lable,1)
%     ml_progressBar(i,50000);
%     sdet=multi_lable(:,i);
%    [~,idx] =sort(sdet);
%    top5=idx(1:5);
%    if find(top5,vallb(i))
%       cor=cor+1; 
%    end
% end



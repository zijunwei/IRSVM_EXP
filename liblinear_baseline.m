
IMGNETinit;

addpath('/nfs/bigbang/zijun/imgnet/imgnet2012/ILSVRC2012_devkit_t12/evaluation');

featdir='/nfs/bigbang/zijun/imgnet/imgnet2012/ExtractedFeatures/cnn_global_imagenet-vgg-verydeep-16_default_aug';

% load annotation:
meta=load('/nfs/bigbang/zijun/imgnet/imgnet2012/ILSVRC2012_devkit_t12/data/meta.mat');
classdict = make_hash(meta.synsets);


% another mapping from 'id' to 1...k (here k=1000) 
% array(i)=id;
multiclslb=zeros(1,IMGNETopts.nclasses);
for i=1:1:IMGNETopts.nclasses
   multiclslb(i)=get_class2node(classdict,train_dirs(i).name);
end

% 1. read all the data...
%training set:
featcell2=cell(1,IMGNETopts.nclasses);
lbcell2=cell(1,IMGNETopts.nclasses);
for i=1:1:IMGNETopts.nclasses
    fprintf('progress : [%d out of %d]\n',i,IMGNETopts.nclasses);
   sdir=fullfile(featdir, IMGNETopts.trainset, train_dirs(i).name);
   featfiles=dir(fullfile(sdir,'*.mat'));
   nfiles=length(featfiles);
   featcell1=cell(1,nfiles);
   for j=1:1:nfiles
       ml_progressBar(j,nfiles);
       glfeat=load(fullfile(sdir,featfiles(j).name));
       featcell1{j}= single(glfeat.gl_feat);
       
   end
   classnode=get_class2node(classdict,train_dirs(i).name);
   k=find(multiclslb==classnode);
   lb=k*ones(1,nfiles);
   lbcell2{i}=lb;
   featmat=cat(2,featcell1{:}); 
   featcell2{i}=featmat;    
end
trd=cat(2,featcell2{:});
trlb=cat(2,lbcell2{:});


% validation set:
val_dir=fullfile(featdir,IMGNETopts.valset);

feat_files=dir(fullfile(val_dir,'*.mat'));
nval=length(feat_files);
featcell=cell(1,nval);
for i=1:1:nval
    ml_progressBar(i,nval,'val_data');
   glfeat=load(fullfile(val_dir,feat_files(i).name));
   featcell{i}=glfeat.gl_feat;
    
    
end

vald=cat(2,featcell{:});
vallbs=textread('/nfs/bigbang/zijun/imgnet/imgnet2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt','%d');
vallbs=vallbs';
vallb=vallbs;
for i=1:1:length(vallbs)
    ml_progressBar(i,length(vallbs));
    vallb(i)=find(multiclslb==vallbs(i));
end

save('imgnetdata','trd','trlb','vald','vallb','multiclslb','-v7.3');



% % test set:
% test_dir=fullfile(featdir,IMGNETopts.testset);
% 
% feat_files=dir(fullfile(test_dir,'*.mat'));
% ntest=length(feat_files);
% featcell=cell(1,ntest);
% testlb=zeros(1,ntest);
% for i=1:1:ntest
%    glfeat=load(fullfile(test_dir,feat_files(i).name));
%    featcell{i}=glfeat.gl_feat;
%     
%     
% end
% 
% testfeat=cat(2,featcell);






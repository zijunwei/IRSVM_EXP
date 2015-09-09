
% function run_irlssvm(feat_dir)
addpath('/home/hzwzijun/MatlabLibs/cplex125/cplex/matlab');
addpath('/home/hzwzijun/MatlabLibs/cplex125/cplex/examples/src/matlab');
addpath(genpath('/home/hzwzijun/imgdetection/Data/VOCdevkit2007'));
addpath(genpath('/home/hzwzijun/MatlabLibs/MyGradFuncs'));
addpath('/home/hzwzijun/MatlabLibs/vlfeat/toolbox');
vl_setup();
VOCinit;
VOCopts.lbset=[VOCopts.datadir,VOCopts.dataset,'/ImageSets/Main/%s.mat'];

 feat_dir='spp_vgg16_featbags_gpu';

% extract features:
[trLb,trD,trBags]=getFeats2(VOCopts,'trainval',feat_dir,'bags',false,'global_only',false);
[tstLb,tstD,tstBags]=getFeats2(VOCopts,'test',feat_dir,'bags',false,'global_only',false);

save_dir=[feat_dir,'-',datestr(date)];
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end
%
lambda_factor=1e-4;
aps = zeros(length(VOCopts.classes), 2);

for kk=1:length(VOCopts.classes)
    fprintf('Running for class %s\n', VOCopts.classes{kk});
    startT0 = tic;
    
    
    trLb_i  = trLb(:,kk);
    trIdxs = trLb_i ~= 0;
    trLb_i   = trLb_i(trIdxs);
    
    tstLb_i  = tstLb(:,kk);
    tstIdxs = tstLb_i ~= 0;
    tstLb_i = tstLb_i(tstIdxs);
    
    
    
    %% baseline
    fprintf('running for baselines \n');
    lambda = lambda_factor*length(trLb_i);
    [w,b] = ML_Ridge.ridgeReg(trD(:,trIdxs), trLb_i, lambda, ones(size(trLb_i)));
    
    initScore = tstD(:,tstIdxs)'*w + b;
    aps(kk,1) = ml_ap(initScore, tstLb_i, 0);
    
    %%  iterative
%     opts.initOpt = 'wb';
%     opts.w = w;
%     opts.b = b;
%     opts.nIter = 100;
%     opts.nThread = 1;
%     
%     opts.methodName = 'IRLSSVM6';
%     opts.initOpt = 'mean';
%     %opts.compactConstrs = {'all'};
%     
%     fprintf('running for irsvm\n');
%     [w, b, s, objVals] = MIR_IRLSSVM6.train(trBags(trIdxs), trLb_i, lambda, opts);
%     tstScore = M_IRSVM.predict(tstBags(tstIdxs), w, b, s);
%     aps(kk,2) = ml_ap(tstScore, tstLb_i, 0);
%     classap=aps(kk,:);
%     fprintf('%s, start: %5.2f, IRLSSVM6: %5.2f\n', ...
%         VOCopts. classes{kk}, 100*classap);
%     fprintf('#non-zero s: %d\n\n\n', sum(s > 1e-3));
%     
%     
%     elapseT = toc(startT0);
%     save(sprintf([save_dir,'/res_%s.mat'],VOCopts.classes{kk}),'w','b','s','objVals','classap');
%     fprintf('Total elapseT: %g\n', elapseT);
end
save('meanAPs.mat','aps');
% end

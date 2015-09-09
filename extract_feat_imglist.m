function extract_feat_imglist(imglist,save_dir,opts)
% feats=cell(length(imglist),1);
for i=1:1:length(imglist)
    [~,save_name,~]=fileparts(imglist{i});
    fprintf('processing %s : [%d out of %d]\n',save_name,i+opts.startid-1,opts.endid);
    im=imread(imglist{i});
    im=convert2Color(im);
    if strcmpi(opts.feattype,'global')
        gl_feat=extract_feat(im,opts);
        save(fullfile(save_dir,save_name),'gl_feat');
    elseif strcmpi(opts.feattype,'local')
        re_feat=extract_local_feat(im,  save_name,  opts);
        save(fullfile(save_dir,save_name),'re_feat');
        
    end
end



end

function re_feat=extract_local_feat(im,save_name,opts)
pp=load(fullfile(opts.impps,[save_name,'.JPEG.mat']));
pboxes=pp.bbs;

[~,sortid]=sort(pboxes(:,end),'descend');
pboxes=pboxes(sortid,:);
if size(pboxes,1)>opts.topN
    
    pboxes =pboxes(1:opts.topN,:);
end
nboxes=size(pboxes,1);
re_featcell=cell(1,nboxes);
for i=1:1:nboxes
    ml_progressBar(i,nboxes);
    sub_im=im(pboxes(i,2):pboxes(i,4),pboxes(i,1):pboxes(i,3),:);
    re_featcell{i}=extract_feat(sub_im,opts);
    
end

re_feat=cat(2,re_featcell{:});

end
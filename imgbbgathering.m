function imgbbgathering
% prerun;
setname='trainval';
VOCinit();
gtids=textread(sprintf(VOCopts.imgsetpath,setname),'%s');
args.startid=1;
args.endid=100;
args.norm=true;



opts.topN=100;




for i=args.startid:args.endid
    fprintf(' processing image : %s  %d \\ [%d - %d] \n', gtids{i},i,args.startid,args.endid);
    
    im=imread(sprintf(VOCopts.imgpath,gtids{i}));
   
    
    
    
    


    % img region proposl
    bbs=[];
    load(sprintf(VOCopts.regproposalpath,gtids{i}));
    if isempty(bbs)
        error('%s doesn''t have corresponding propsoal regios\n ')
    end
    
    [~,o]=sort(bbs(:,end),'descend');
    pboxes=bbs(o,1:4);
    
    if length(pboxes)>opts.topN;
        pboxes=pboxes(1:opts.topN,:);
    end
    
    
   
    
    debuginfor(i).im=im;
    debuginfor(i).pboxes=pboxes;
    
    
end



save('debugval.mat','debuginfor','-v7.3');



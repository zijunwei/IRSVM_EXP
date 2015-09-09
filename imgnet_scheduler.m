function imgnet_scheduler(mode)
imgnetOpts=imgnet_gl_initialization();


f_handle=@imgnet_cnn_vgg16_global_feat_extraction_dirbased;
if mode==1 % train first half
    for i=104:1:500
        fprintf('processing ... [ %s ] \n\n\n',fullfile('ILSVRC2012_img_train',imgnetOpts.wnids{i}));
        f_handle( fullfile('ILSVRC2012_img_train',imgnetOpts.wnids{i}));
        
    end
    
    
elseif mode==2  % train second  half
    
    for i=501:1:1000
        fprintf('processing ... [ %s ] \n\n\n',fullfile('ILSVRC2012_img_train',imgnetOpts.wnids{i}));
        f_handle( fullfile('ILSVRC2012_img_train',imgnetOpts.wnids{i}));
        
    end
    
    
elseif mode==3  %  val set
    
    f_handle( ('ILSVRC2012_img_val'));
    
elseif mode==4 % test set
     f_handle( ('ILSVRC2012_img_test'));
end

end

function imgnetOpts=imgnet_gl_initialization()
imgnetOpts.root='/nfs/bigbang/zijun/imgnet/imgnet2012';

imgnetOpts.train_dir=[imgnetOpts.root,'/ILSVRC2012_img_train'];
imgnetOpts.val_dir=[imgnetOpts.root,'/ILSVRC2012_img_val'];
imgnetOpts.glfeat_dir=[imgnetOpts.root,'/ILSVRC2012_glfeat_cnn'];
train_dirs=dir(fullfile(imgnetOpts.train_dir,'n*'));
imgnetOpts.wnids={train_dirs.name};

end

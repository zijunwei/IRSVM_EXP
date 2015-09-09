clear IMGNETopts


IMGNETopts.dataset='Imgnet2012';

IMGNETopts.root='/nfs/bigbang/zijun/imgnet/imgnet2012';

IMGNETopts.train_dir=[IMGNETopts.root,'/ILSVRC2012_img_train'];
IMGNETopts.val_dir=[IMGNETopts.root,'/ILSVRC2012_img_val'];
IMGNETopts.test_dir=[IMGNETopts.root,'/ILSVRC2012_img_test'];

IMGNETopts.trainset='ILSVRC2012_img_train';
IMGNETopts.valset='ILSVRC2012_img_val';
IMGNETopts.testset='ILSVRC2012_img_test';

train_dirs=dir(fullfile(IMGNETopts.train_dir,'n*'));
IMGNETopts.wnids={train_dirs.name};
IMGNETopts.nclasses=length(train_dirs);


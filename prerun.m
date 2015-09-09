% matlab vl_setup
run('matlab/vl_setupnn.m');

% add vl-feat
addpath('/home/hzwzijun/MatlabLibs/vlfeat/toolbox');
vl_setup();


 % add voc devkit path
addpath(genpath('/home/hzwzijun/imgdetection/Data/VOCdevkit2007'));

% add spm_pool
addpath('spm_pool');
addpath('bin')
% addpath('matlab')

% add minh's 'MyGradFunction'
addpath(genpath('/home/hzwzijun/MatlabLibs/MyGradFuncs'));

addpath('./utils');

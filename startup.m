function startup()
addpath spm_pool
addpath bin

if ~exist('spm_pool_caffe_mex.mexa64','file')
    fprintf('Compiling spm_pool_caffe_mex and save in bin \n');
    
    mex -outdir bin ...
        -largeArrayDims ...
        spm_pool/spm_pool_caffe_mex.cpp ...
        -output spm_pool_caffe_mex;
end
%% add the matconvnet compiles

cuda7_0path='/usr/local/cuda';
% cuda6_5path='/usr/local/cuda';

addpath matlab
vl_compilenn('enableGpu', true, 'cudaRoot', cuda7_0path);

% if  exist(cuda7_0path,'dir')
%     vl_compilenn('enableGpu', true, 'cudaRoot', cuda7_0path)
% else
%     vl_compilenn('enableGpu', true, 'cudaRoot', cuda6_5path)
% end

end

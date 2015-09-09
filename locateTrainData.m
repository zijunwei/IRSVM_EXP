%given the name of the directory, find the location of in imagenet.train
%data.

function idx=locateTrainData(dirname,IMGNETopts)
if nargin<2
   IMGNETopts=[]; 
end
if isempty(IMGNETopts)
 IMGNETinit; 
end

idxcell=strfind(IMGNETopts.wnids,dirname);
idx=find(not(cellfun('isempty', idxcell)));


end
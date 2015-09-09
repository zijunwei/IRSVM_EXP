%% get top 5 error
cortop5=0;
cortop1=0;
%multi_lable=multi_models.w*vald;
for i=1:1:size(multi_lable,2)
    ml_progressBar(i,50000);
    sdet=multi_lable(:,i);
   [~,idx] =sort(sdet,'descend');
   top5=idx(1:5);
   if find(top5==vallb(i))
      cortop5=cortop5+1; 
   end
    if find(top5(1)==vallb(i))
      cortop1=cortop1+1; 
   end
end
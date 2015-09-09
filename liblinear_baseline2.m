%save('imgnetdata','trd','trlb','vald','vallb','multiclslb','-v7.3');


addpath('liblinear/matlab/')
trdsparse=sparse(double( trd));
multi_models=liblineartrain(trlb',(trdsparse),' -s 4 ','col');
[predict_label, accuracy, dec_values] = liblinearpredict(vallb', double(vald), multi_models,['col']);

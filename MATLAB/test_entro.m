clear;  clc
close all; 
 
 
C = 1;
num = 15;
mc_times = 50;  % Total number of Monte-Carlo trials for each alphabet size 
record_S = ceil(logspace(2, 7, num));
record_n = ceil(C*record_S./log(record_S));

twonum = rand(2,1); 
for iter = num:-1:1 
    S = record_S(iter)
    n = record_n(iter)
    dist = betarnd(twonum(1),twonum(2),S,1);
    dist = dist/sum(dist);                    
%     dist = repmat(1/S,S,1);                          
    true_S(iter) = entropy_true(dist);     
    samp = randsmpl(dist, n, mc_times, 'int32');    
	record_JVHW = est_entro_JVHW(samp);       
	record_MLE = est_entro_MLE(samp);  
    JVHW_err(iter) = mean(abs(record_JVHW - true_S(iter))); 
    MLE_err(iter) = mean(abs(record_MLE - true_S(iter)));    
end 

figure(1) 
plot(record_S./record_n, JVHW_err,'b-s','LineWidth',2,'MarkerFaceColor','b'); hold on;
plot(record_S./record_n, MLE_err,'r-.o','LineWidth',2,'MarkerFaceColor','r'); 
legend('JVHW','MLE');
xlabel('S/n')
ylabel('Mean Absolute Error')
title('Entropy Estimation')
xlim([4, 16.5])  
 


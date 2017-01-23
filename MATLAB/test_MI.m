clear; close all; clc
 
% This demo tests the performance of the mutual information estimation for 
% a specific discrete transition channel model 
%                         Y = (X + Z) mod S,
% where X ~ px, Z ~ pz, and X and Z are independent and both supported on 
% {0, 1, ..., S-1}.

num = 8;
mc_times = 300;   % The number of Monte-Carlo trials for each single S
record_S = round(logspace(2, log10(200), num));
n = 2500;        % The operating regime is: S << n << S^2
true_S = zeros(1,num);
JVHW_S = zeros(1,num);
MLE_S = zeros(1,num);
for iter = num:-1:1
    S = record_S(iter)
    px = betarnd(0.6,0.5,S,1);
    px = px/sum(px);  
    pz = betarnd(0.6,0.5,S,1);
    pz = pz/sum(pz);          
    py_cond_x = bsxfun(@circshift, pz, 0:S-1).';
    pxy = diag(px)*py_cond_x;
    [X,Y] = ind2sub([S,S],randsmpl(pxy(:),n, mc_times));
    true_S(iter) = MI_true(pxy);
	record_JVHW = est_MI_JVHW(X,Y);      
	record_MLE = est_MI_MLE(X,Y);      
    JVHW_S(iter) = mean((record_JVHW - true_S(iter)).^2);  
    MLE_S(iter) = mean((record_MLE - true_S(iter)).^2);     
end

semilogy(log(record_S), JVHW_S,'b-s','LineWidth',2,'MarkerFaceColor','b');
hold on;
semilogy(log(record_S), MLE_S,'r-.o','LineWidth',2,'MarkerFaceColor','r');
legend('JVHW estimator','MLE');
xlabel('log S')
ylabel('Mean Squared Error')
title('Mutual Information Estimation')
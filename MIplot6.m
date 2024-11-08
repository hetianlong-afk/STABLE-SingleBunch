clc;clear;
I_rank = [0.001,0.1,0.2:0.2:4]*1442; 
sigmap=zeros(1,length(I_rank));sigmaq=zeros(1,length(I_rank));
meanp=zeros(1,length(I_rank));meanq=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
    filename=['.\STCF-2GeV\CSR+CWR_0.05mm_Dq0.1ps_bin0.002ps_2MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmae = HALF.sigma_e0;
    sigmap(i)=std(record_P_std(1e3:end))*sigmae;meanp(i)=mean(record_P_std(1e3:end))*sigmae;
    sigmaq(i)=std(record_Q_std(1e3:end))*10;meanq(i)=mean(record_Q_std(1e3:end))*10;
end
%
sigmap1=zeros(1,length(I_rank));sigmaq1=zeros(1,length(I_rank));
meanp1=zeros(1,length(I_rank));meanq1=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
    filename=['.\STCF-2GeV\CSR+CWR_0.05mm_Dq0.1ps_bin0.002ps_10MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmap1(i)=std(record_P_std(3e3:end))*sigmae;meanp1(i)=mean(record_P_std(3e3:end))*sigmae;
    sigmaq1(i)=std(record_Q_std(3e3:end))*10;meanq1(i)=mean(record_Q_std(3e3:end))*10;
end
% 
sigmap2=zeros(1,length(I_rank));sigmaq2=zeros(1,length(I_rank));
meanp2=zeros(1,length(I_rank));meanq2=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
    filename=['.\STCF-2GeV\CSR+CWR_0.05mm_Dq0.1ps_bin0.002ps_20MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmap2(i)=std(record_P_std(3e3:end))*sigmae;meanp2(i)=mean(record_P_std(3e3:end))*sigmae;
    sigmaq2(i)=std(record_Q_std(3e3:end))*10;meanq2(i)=mean(record_Q_std(3e3:end))*10;
end

sigmap3=zeros(1,length(I_rank));sigmaq3=zeros(1,length(I_rank));
meanp3=zeros(1,length(I_rank));meanq3=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
    filename=['.\STCF-2GeV\CSR+CWR_0.05mm_Dq0.1ps_bin0.002ps_50MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmap3(i)=std(record_P_std(3e3:end))*sigmae;meanp3(i)=mean(record_P_std(3e3:end))*sigmae;
    sigmaq3(i)=std(record_Q_std(3e3:end))*10;meanq3(i)=mean(record_Q_std(3e3:end))*10;
end
% 
sigmap4=zeros(1,length(I_rank));sigmaq4=zeros(1,length(I_rank));
meanp4=zeros(1,length(I_rank));meanq4=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
    filename=['.\STCF-2GeV\CSR+CWR_0.2mm_10MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmap4(i)=std(record_P_std(2e3:end))*sigmae;meanp4(i)=mean(record_P_std(2e3:end))*sigmae;
    sigmaq4(i)=std(record_Q_std(2e3:end))*10;meanq4(i)=mean(record_Q_std(2e3:end))*10;
end
% % 
sigmap5=zeros(1,length(I_rank));sigmaq5=zeros(1,length(I_rank));
meanp5=zeros(1,length(I_rank));meanq5=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
    filename=['.\STCF-2GeV\CSR+CWR_0.2mm_20MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmap5(i)=std(record_P_std(2e3:end))*sigmae;meanp5(i)=mean(record_P_std(2e3:end))*sigmae;
    sigmaq5(i)=std(record_Q_std(2e3:end))*10;meanq5(i)=mean(record_Q_std(2e3:end))*10;
end
% % 
% sigmap6=zeros(1,length(I_rank));sigmaq6=zeros(1,length(I_rank));
% meanp6=zeros(1,length(I_rank));meanq6=zeros(1,length(I_rank));
% for i=1:length(I_rank)
%     I0=I_rank(i);
%     filename=['.\有CSR_CASE4\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
%     load(filename);
%     sigmap6(i)=std(record_P_std(2e3:end))*7.44e-4;meanp6(i)=mean(record_P_std(2e3:end))*7.44e-4;
%     sigmaq6(i)=std(record_Q_std(2e3:end))*10;meanq6(i)=mean(record_Q_std(2e3:end))*10;
% end
% 
% sigmap7=zeros(1,length(I_rank));sigmaq7=zeros(1,length(I_rank));
% meanp7=zeros(1,length(I_rank));meanq7=zeros(1,length(I_rank));
% for i=1:length(I_rank)
%     I0=I_rank(i);
%     filename=['.\有CSR_CASE4\MIGPU_n_3_I_tot_',num2str(I0),'mA.mat'];
%     load(filename);
%     sigmap7(i)=std(record_P_std(2e3:end))*7.44e-4;meanp7(i)=mean(record_P_std(2e3:end))*7.44e-4;
%     sigmaq7(i)=std(record_Q_std(2e3:end))*10;meanq7(i)=mean(record_Q_std(2e3:end))*10;
% end


%%
clc;clear;
I_rank = [0.001,0.1,0.2:0.2:3]*1442; 

sigmap=zeros(1,length(I_rank));sigmaq=zeros(1,length(I_rank));
meanp=zeros(1,length(I_rank));meanq=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
    filename=['MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmae = HALF.sigma_e0;
    sigmap(i)=std(record_P_std(3e3:end))*sigmae;meanp(i)=mean(record_P_std(3e3:end))*sigmae;
    sigmaq(i)=std(record_Q_std(3e3:end))*10;meanq(i)=mean(record_Q_std(3e3:end))*10;
end
%%

figure(1)
subplot(2,1,1);
% figure
% errorbar(I_rank/1442,meanp,sigmap,'bo','Linewidth',1.0);hold on;
% errorbar(I_rank/1442,meanp1,sigmap1,'go','Linewidth',1.0);hold on;
% errorbar(I_rank/1442,meanp2,sigmap2,'ro','Linewidth',1.0);hold on;
% errorbar(I_rank/1442,meanp3,sigmap3,'ko','Linewidth',1.0);hold on;
errorbar(I_rank/1442,meanp4,sigmap4,'b*:','Linewidth',1.0);hold on;
errorbar(I_rank/1442,meanp5,sigmap4,'m*:','Linewidth',1.0);hold on;
xlabel('I [mA]');
ylabel('\sigma_{\delta}');
% legend('wo HHC @NEG1um','wo HHC @NEG2.5um','w HHC @NEG1um','w HHC @NEG2.5um');
set(gca,'FontName','Times New Roman','FontSize',12);

subplot(2,1,2);
% errorbar(I_rank/1442,meanq,sigmaq,'bo','Linewidth',1.0);hold on;
% errorbar(I_rank/1442,meanq1,sigmaq1,'go','Linewidth',1.0);hold on;
% errorbar(I_rank/1442,meanq2,sigmaq2,'ro','Linewidth',1.0);hold on;
% errorbar(I_rank/1442,meanq3,sigmaq3,'ko','Linewidth',1.0);hold on;
errorbar(I_rank/1442,meanq4,sigmaq4,'b*:','Linewidth',1.0);hold on;
errorbar(I_rank/1442,meanq5,sigmaq5,'m*:','Linewidth',1.0);hold on;
% errorbar(I_rank/800,meanq5,sigmaq5,'r.:','Linewidth',1.0);hold on;

% errorbar(I_rank/800,meanq6,sigmaq6,'ko-','Linewidth',1.5);hold on;
% errorbar(I_rank/800,meanq7,sigmaq7,'ko:','Linewidth',1.5);hold on;
xlabel('I [mA]');
ylabel('\sigma_t [ps]');
% legend('wo HHC RW(1um)+Geo.','wo HHC RW(1um)+Geo.+CSR','w HHC RW(1um)+Geo.','w HHC RW(1um)+Geo.+CSR',...
%     'wo HHC RW(2.5um)+Geo.','wo HHC RW(2.5um)+Geo.+CSR','w HHC RW(2.5um)+Geo.','w HHC RW(2.5um)+Geo.+CSR');
% ylim([34,56]);
set(gca,'FontName','Times New Roman','FontSize',12);

%%
% subplot(2,1,1);legend('RMS0.05mm @10M','RMS0.05mm @20M','RMS0.05mm @50M','RMS0.2mm @10 M','RMS0.2mm @20 M');
subplot(2,1,1);legend('RMS0.2mm @10 M','RMS0.2mm @20 M');
% subplot(1,2,2);legend('RMS0.05mm @10M','RMS0.05mm @20M','RMS0.05mm @50M','RMS0.2mm @10 M','RMS0.2mm @20 M');
%%
subplot(1,2,1);legend('rms 0.2 mm','rms 0.1 mm','rms 0.05 mm','rms 0.02 mm','rms 0.01 mm');
subplot(1,2,2);legend('rms 0.2 mm','rms 0.1 mm','rms 0.05 mm','rms 0.02 mm','rms 0.01 mm');
subplot(1,2,1);legend('NEG 0.5um','NEG 1.0um','NEG 1.5um');
subplot(1,2,2);legend('NEG 0.5um','NEG 1.0um','NEG 1.5um');
subplot(1,2,1);legend('w/o CSR','w/ CSR');
subplot(1,2,2);legend('w/o CSR','w/ CSR');
subplot(1,2,1);legend('\tau_z=2 ms','\tau_z=14 ms');
subplot(1,2,2);legend('\tau_z=2 ms','\tau_z=14 ms');

subplot(1,2,1);legend('5M & 0.1ps','5M & 0.05ps','5M & 0.02ps','10M & 0.02ps');
subplot(1,2,2);legend('5M & 0.1ps','5M & 0.05ps','5M & 0.02ps','10M & 0.02ps');

subplot(1,2,1);legend('rms 0.1 mm','rms 0.05 mm','rms 0.02 mm','rms 0.01 mm','rms 0.002 mm');
subplot(1,2,2);legend('rms 0.1 mm','rms 0.05 mm','rms 0.02 mm','rms 0.01 mm','rms 0.002 mm');

subplot(1,2,1);legend('rms 0.1 mm','rms 0.01 mm','rms 0.002 mm','rms 0.001 mm');
subplot(1,2,2);legend('rms 0.1 mm','rms 0.01 mm','rms 0.002 mm','rms 0.001 mm');

subplot(1,2,1);legend('2M & 0.02 ps','5M & 0.02 ps','5M & 0.01 ps');
subplot(1,2,2);legend('2M & 0.02 ps','5M & 0.02 ps','5M & 0.01 ps');
%%
plot(bin_num_sumq_gaussian);hold on;

%%
subplot(1,2,1);
% legend('SSL:wO HHC','SSL: with HHC','SSL+5um Cu: wo HHC','SSL+5um Cu: with HHC','Al: wo HHC','Al: with HHC','CuCrZr+1umNEG: wO HHC','CuCrZr+1umNEG: with HHC');

legend('RW+Geo*2+CSR','RW+Geo*2+CSR+HHC','RW+Geo*2','RW+Geo*2+HHC');

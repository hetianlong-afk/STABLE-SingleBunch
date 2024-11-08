clc;clear;
I_rank = [0.001,0.1,0.2:0.2:4]*1442; 
sigmap=zeros(1,length(I_rank));sigmaq=zeros(1,length(I_rank));
meanp=zeros(1,length(I_rank));meanq=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
     filename=['.\STCF-2GeV\CSR+CWR_0.2mm_10MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
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
     filename=['.\STCF-2GeV\CSR+CWR_0.2mm_20MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmap1(i)=std(record_P_std(3e3:end))*sigmae;meanp1(i)=mean(record_P_std(3e3:end))*sigmae;
    sigmaq1(i)=std(record_Q_std(3e3:end))*10;meanq1(i)=mean(record_Q_std(3e3:end))*10;
end
% 
I_rank = [0.001,0.1:0.05:0.4,0.5:0.1:1]*1442;
sigmap2=zeros(1,length(I_rank));sigmaq2=zeros(1,length(I_rank));
meanp2=zeros(1,length(I_rank));meanq2=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
     filename=['.\STCF-1GeV-rightWP\CSR+CWR_0.2mm_10MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmap2(i)=std(record_P_std(3e3:end))*sigmae;meanp2(i)=mean(record_P_std(3e3:end))*sigmae;
    sigmaq2(i)=std(record_Q_std(3e3:end))*10;meanq2(i)=mean(record_Q_std(3e3:end))*10;
end
I_rank = [0.001,0.1:0.05:0.4,0.5:0.1:1]*1442;
sigmap3=zeros(1,length(I_rank));sigmaq3=zeros(1,length(I_rank));
meanp3=zeros(1,length(I_rank));meanq3=zeros(1,length(I_rank));
for i=1:length(I_rank)
    I0=I_rank(i);
     filename=['.\STCF-1GeV-rightWP\CSR+CWR_0.2mm_20MP\MIGPU_n_0_I_tot_',num2str(I0),'mA.mat'];
    load(filename);
    sigmap3(i)=std(record_P_std(3e3:end))*sigmae;meanp3(i)=mean(record_P_std(3e3:end))*sigmae;
    sigmaq3(i)=std(record_Q_std(3e3:end))*10;meanq3(i)=mean(record_Q_std(3e3:end))*10;
end
% 

%%

figure(1)
subplot(2,1,1);
title('2 GeV');
% figure
yyaxis left
I_rank = [0.001,0.1,0.2:0.2:4]*1442; 
% errorbar(I_rank/1442,meanp,sigmap,'bo:','Linewidth',1.0);hold on;
errorbar(I_rank/1442,meanp1,sigmap1,'b*:','Linewidth',1.0);hold on;
ylim([7e-4,10e-4]);
ylabel('\sigma_{\delta}');
set(gca,'FontName','Times New Roman','FontSize',12);
yyaxis right
% errorbar(I_rank/1442,meanq,sigmaq,'ro:','Linewidth',1.0);hold on;
errorbar(I_rank/1442,meanq1,sigmaq1,'ro:','Linewidth',1.0);hold on;
ylabel('\sigma_t [ps]');
set(gca,'FontName','Times New Roman','FontSize',12);
I_rank = [0.001,0.1:0.05:0.4,0.5:0.1:1]*1442;


subplot(2,1,2);
title('1 GeV');
yyaxis left
% errorbar(I_rank/1442,meanp2,sigmap2,'bo:','Linewidth',1.0);hold on;
errorbar(I_rank/1442,meanp3,sigmap3,'b*:','Linewidth',1.0);hold on;
ylabel('\sigma_{\delta}');
ylim([7e-4,13e-4]);
yyaxis right
% errorbar(I_rank/1442,meanq2,sigmaq2,'ro:','Linewidth',1.0);hold on;
errorbar(I_rank/1442,meanq3,sigmaq3,'ro:','Linewidth',1.0);hold on;

xlabel('I [mA]');
ylabel('\sigma_t [ps]');
% legend('wo HHC @NEG1um','wo HHC @NEG2.5um','w HHC @NEG1um','w HHC @NEG2.5um');
set(gca,'FontName','Times New Roman','FontSize',12);

% subplot(2,1,2);
% I_rank = [0.001,0.1,0.2:0.2:4]*1442; 
% errorbar(I_rank/1442,meanq,sigmaq,'bo:','Linewidth',1.0);hold on;
% errorbar(I_rank/1442,meanq1,sigmaq1,'b*:','Linewidth',1.0);hold on;
% I_rank = [0.001,0.1:0.05:0.4,0.5:0.1:1]*1442; 
% errorbar(I_rank/1442,meanq2,sigmaq2,'ro:','Linewidth',1.0);hold on;
% errorbar(I_rank/1442,meanq3,sigmaq3,'g*:','Linewidth',1.0);hold on;
% xlabel('I [mA]');
% ylabel('\sigma_t [ps]');
% legend('wo HHC RW(1um)+Geo.','wo HHC RW(1um)+Geo.+CSR','w HHC RW(1um)+Geo.','w HHC RW(1um)+Geo.+CSR',...
%     'wo HHC RW(2.5um)+Geo.','wo HHC RW(2.5um)+Geo.+CSR','w HHC RW(2.5um)+Geo.','w HHC RW(2.5um)+Geo.+CSR');
% ylim([34,56]);
set(gca,'FontName','Times New Roman','FontSize',12);

%%
% subplot(2,1,1);legend('RMS0.05mm @10M','RMS0.05mm @20M','RMS0.05mm @50M','RMS0.2mm @10 M','RMS0.2mm @20 M');
subplot(2,1,1);legend('1 GeV & 10 M','1 GeV & 20 M');
% subplot(1,2,2);legend('RMS0.05mm @10M','RMS0.05mm @20M','RMS0.05mm @50M','RMS0.2mm @10 M','RMS0.2mm @20 M');
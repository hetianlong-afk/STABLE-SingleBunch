% 单束团纵向追踪
% 研究单束团纵向运动
% 纵向电阻壁尾场
% GPU 加速 500e4 particles
% 作者： 何天龙
% 时间： 20200927
% 代入电阻壁尾势数据 modified at 2020/10/22 by tianlong
clc;clear;
%% beam parameters
% STCF @ 2GeV
% C        = 865.398;       % 周长 m
% I0       = 2000e-3;      % 流强 A
% E0       = 2.0e9;       % 能量 eV
% U0       = 541e3;       % 能损 eV 500e3 186.5e3
% sigma_t0 = 10e-12;      % s  束团归一化长度 （用于计算上归一化）
% sigma_e0 = 7.9e-4;      %    束团归一化能散  6.16e-4  6.44e-4
% alpha_c  = 13.86e-4;     % 动量紧缩因子 18.86e-4 -> 13.86E-4
% tau_s    = 10.67e-3;     % 阻尼时间 s 可调  取较小的值可加速收敛 23.4
% tau_z    = 10.67e-3;
% V_mc     = 3e6;       % 主腔腔压 1.1e6
% h        = 1442;         % 谐波数
% n_hc     = 0;           % 3-w HHC ; 0- w/o HHC for STCF

% STCF @ 1GeV
C        = 865.398;       % 周长 m
I0       = 1500e-3;      % 流强 A
E0       = 1.0e9;       % 能量 eV
U0       = 105.9e3;       % 能损 eV 500e3 186.5e3
sigma_t0 = 10e-12;      % s  束团归一化长度 （用于计算上归一化）
sigma_e0 = 5.61e-4;      %    束团归一化能散  6.16e-4  6.44e-4
alpha_c  = 13.04e-4;     % 动量紧缩因子 6.3e-5
tau_s    = 27.26e-3;     % 阻尼时间 s 可调  取较小的值可加速收敛 23.4
tau_z    = 27.26e-3;
V_mc     = 1e6;       % 主腔腔压 1.1e6
h        = 1442;         % 谐波数
n_hc     = 0;           % 3-w HHC ; 0- w/o HHC for STCF



% HHC Param.
Q_hc      = 2e8;        % 品质因素
R_hc      = 2e8 * 41;   % 2 cavities 特征阻抗
fre_shift = 58e3;       % 60e3 3 倍；72e3 2倍
% fre_shift = detune_HC_calc(I0,3,C,h,U0,V_mc,R_hc,Q_hc);     % 失谐频率 Hz
%% fill pattern
pattern  = zeros(1,h);
pattern(1:1:h)=1;
fillrate = length(find(pattern==1))/h;

%% 循环 扫描流强 I0
% I_ave = [40:40:800,900:100:1600,1800:200:6400]*1e-3;   % 失谐频率 Hz
% I_ave = [4,40:40:800]*1e-3;
% I_ave = [280:40:400]*1e-3;
% I_ave = [20:20:300]*1e-3; 
% I_ave = [0.001,0.1,0.2:0.2:4]*1442*1e-3; 
% I_ave = [0.001,0.1:0.05:0.4,0.5:0.1:1]*1442*1e-3; 
I_ave = [1]*1442*1e-3;
% I_ave = [4.0]*1442*1e-3; 
n_hc_scan = [0];     % 0 or 3
FPCondition = 0;        % flat potential condition
for ni = 1:length(n_hc_scan)
n_hc =  n_hc_scan(ni);   
for Ii = 1:length(I_ave)

I_tot = I_ave(Ii);
HALF = machine(C,I0,I_tot,U0,E0,tau_s,tau_z,sigma_t0,sigma_e0,alpha_c,h,V_mc,...
    n_hc,R_hc,Q_hc,fillrate,fre_shift,FPCondition);

%% bunch generation
% 将单个束团分成 500份，每份1万个粒子
% 等效于 500个束团，每个束团1万个粒子
Par_num = 10e3; 
Bun_num = 2000;
Total_num = Par_num * Bun_num;
q = zeros(Par_num,Bun_num);p = zeros(Par_num,Bun_num);
for i=1:Bun_num
    q(:,i)=TruncatedGaussian(1, [-3,3], [Par_num,1]);
    p(:,i)=TruncatedGaussian(1, [-3,3], [Par_num,1]);
end
% CPU to GPU
Q=single(gpuArray(q)); P=single(gpuArray(p));     % single type

%% wake data (intrabunch motion)
Dq = 0.01;
tau_q = (0:Dq:100)'*sigma_t0;
%% Total RW wake
% new method : 2020/10/07
% binwidth = 0.01 ps or 0.002 ps 0.001 ps 0.0005 ps  0.0002 ps

binwidth_rw = 0.002 * 1e-12; % s
tau_q_rw_right = (0:binwidth_rw:5e-10)'; % -300 ps 到 300 ps
tau_q_rw_left = (0:-binwidth_rw:-5e-10)';
Wake_inter = zeros(length(tau_q_rw_right),1);
%%
% load('.\HALF_RW0.2mm\HALF_RW_wpz.mat');wakelong_rw=wakepz;
% % plot(t,wakepz,'LineWidth',1.5);xlabel('t [s]');ylabel('long.rw wake [V/C]');set(gca,'FontName','Times New Roman','FontSize',12);hold on;
% 
% load('HALF_RW_wpz.mat');
% wakelong_rw=wakepz; % *2/3
% figure(199)
% plot(t,wakepz,'LineWidth',1.5);xlabel('t [s]');ylabel('Long.RW wakepotential [V/C]');set(gca,'FontName','Times New Roman','FontSize',12);hold on;

% 0.2 mm rms CSR wakepotential
% % load('.\HALF_CSR\BendMagnet_CSR_WP.mat');wakelong_bmcsr=wakeplong;
% % plot(t,wakeplong,'b','LineWidth',1.5);xlabel('t [s]');ylabel('long.rw wake [V/C]');set(gca,'FontName','Times New Roman','FontSize',12);hold on;
% 
load('STCF0.2mm_CWRPP1GeV_wpz.mat');
wakelong_dwcwr=wakepz;

load('STCF0.2mm_CSRPP_wpz.mat');
wakelong_bmcsr=wakepz;
% tau=-5e-10:1e-14:5e-10; wakepz = interp1(t',wakeplong',tau);
% wakeplong = wakepz';
% t = tau';
% wakelong_bmcsr=wakeplong;
% plot(t,wakepz,'r','LineWidth',1.5);xlabel('t [s]');ylabel('Longitudinal wakepotential [V/C]');set(gca,'FontName','Times New Roman','FontSize',12);
% % % % 
% wakeplong = wakelong_rw + wakelong_bmcsr;
% plot(t,wakeplong,'LineWidth',1.5);legend('RW','CSR','RW+CSR');

wakeplong = wakelong_bmcsr+wakelong_dwcwr;
% wakeplong = wakelong_rw;
%%
Wake_rw_right = interp1(t,wakeplong,tau_q_rw_right);
Wake_rw_left  = interp1(t,wakeplong,tau_q_rw_left);
Wake_inter_right = Wake_inter + Wake_rw_right;
Wake_inter_left  = Wake_inter + Wake_rw_left;

% Wake_inter_right = Wake_inter;
% Wake_inter_left  = Wake_inter;
%% Total Geometry wake % 0.5 mm rms bunch length
% load('.\HALF_Geo\Geometry_long_wp.mat');
% % plot(t,wakepz,'b','LineWidth',1.5);xlabel('t [s]');ylabel('wake V/C');grid minor;title('long.geo. wake');
% Wake_geo_right = interp1(t,wakepz*2,tau_q_rw_right); % scale a factor of 2
% Wake_geo_left  = interp1(t,wakepz*2,tau_q_rw_left); 
% 
% Wake_inter_right = Wake_inter + Wake_rw_right + Wake_geo_right;
% Wake_inter_left  = Wake_inter + Wake_rw_left +Wake_geo_left;

%% BBR wake
% fr = 586e9;Qs=3.27;Rs = 2e4*1.1*Qs;rot_coef = sqrt(1-1/4/(Qs^2));
% fr = 15e9;Qs=220;Rs = 32*220;rot_coef = sqrt(1-1/4/(Qs^2));
% tau_q_rw = tau_q_rw_right;
% Wake_BBR = 2*pi*fr*Rs/Qs*exp(-tau_q_rw*2*pi*fr/2/Qs).*(cos(tau_q_rw*2*pi*fr*rot_coef)-sin(tau_q_rw*2*pi*fr*rot_coef)/2/rot_coef/Qs);
% Wake_BBR(1) = Wake_BBR(1)/2;
% plot(tau_q_rw,Wake_BBR);
% 
% Wake_inter_right = Wake_inter_right-Wake_BBR; % note '-'

%%
tau_q             = single(gpuArray(tau_q));
Wake_inter_right  = single(gpuArray(Wake_inter_right));
Wake_inter_left   = single(gpuArray(Wake_inter_left));

%%
% charge per macro-particle   : HALF.qc
HALF.qc        = pattern * HALF.qc / Par_num / Bun_num;        %由单个元素变为一行矩阵
wake_kick_coef = HALF.qc(1) * HALF.kick_coef;

Wake_inter_add_kickcoef_right = Wake_inter_right * wake_kick_coef;% 
Wake_inter_add_kickcoef_left = Wake_inter_left * wake_kick_coef;% 
%% start tracking

Track_num  = 4e4;

% record parameters 
Recor_step = 10;
Recor_num  = Track_num / Recor_step;
record_Q_mean = zeros(Recor_num,1);record_Q_std = zeros(Recor_num,1);
record_P_mean = zeros(Recor_num,1);record_P_std = zeros(Recor_num,1);

record_th = 0;
%%
gd = gpuDevice(); 
tic;
for i =1:Track_num
%% drift
    Q = Q + P * HALF.drift_coef;   
    Q_min = min(min(Q));  %  minimum Q    
    Q_new_rw = Q - Q_min; %  用于 rw wake kick 插值
    Q_new = round(Q_new_rw *(1/Dq));    
    % main cavity kick   + V_mc_kick
    
    % count bins
    binnum=max(max(Q_new))+2; 
    binnum=gather(binnum);
    bin_num_q=BinNumCalZ1(binnum,Q_new); % threadsperblock = 1
    bin_num_q=reshape(bin_num_q,binnum,Bun_num);
    bin_num_sumq = sum(bin_num_q,2);
    
    % gaussian 拟合 （分布光滑处理）
    bin_num_sumq_gaussian = smoothdata(bin_num_sumq,'gaussian',11);% default 11
    % tau_bin                                   % tau_q binsize = 0.1 ps
    tau_bin = (0:binwidth_rw:tau_q(binnum))';   % binwidth_rw = 0.002 ps 
    Q_bin   = tau_bin/HALF.sigma_t0;
    tau_bin_len = length(tau_bin);
    % 插值 计算 分布
    bin_num_sumq_gaussian = interp1(tau_q(1:binnum),bin_num_sumq_gaussian,tau_bin);

    % 归一化处理拟合分布
    bin_num_sumq_gaussian = bin_num_sumq_gaussian*(Total_num/sum(bin_num_sumq_gaussian));

    if mod(i,1000)==0
        disp([num2str(i)])
        figure(3);
        plot(bin_num_sumq_gaussian);
        % if i>=2e4 
            figure(500);
            phase_plot(Q,P,Bun_num,Par_num,sigma_t0,sigma_e0,i,6);
        %     filename = ['Fig_',num2str(i),'.png'];
        %     figure(500);saveas(gcf,filename);
        % end
    end
   
    % 用快速傅里叶变换计算卷积
    a = bin_num_sumq_gaussian;
    % 将尾势的左右两部分拼接在一起
    b = [flip(Wake_inter_add_kickcoef_left(2:tau_bin_len));Wake_inter_add_kickcoef_right(1:tau_bin_len)];

    l=numel(a)+numel(b)-1;n=2.^nextpow2(l);
    kick_conv = real(ifft(fft(a,n).*fft(b,n)));
    
    % 取中间“tau_bin_len”个数
    kick_conv = kick_conv(tau_bin_len:tau_bin_len*2-1); 
    % test 
%     if mod(i,10)==0
%     figure(5);plot(tau_bin,kick_conv);hold on;
%     end
    %插值计算每个粒子的wake_kick
    wake_kick = interp1(Q_bin,kick_conv,Q_new_rw)*min(1,i/5000);
    % 统计相空间分布
%     % convolution    
%     kick_conv = conv(bin_num_sumq,Wake_inter_add_kickcoef(1:length(tau_bin)));
%     kick_conv = kick_conv(1:binnum);
%     % wake kick    + wake_kick
%     wake_kick = kick_conv(Q_new+1); % modified in 2020/09/22

    % radiation damping and quantum excitation term  + wake_kick
    P = P + wake_kick;
% main cavity kick   + V_mc_kick
    V_mc_kick   = HALF.rfcoef1*sin(HALF.rfcoef2*Q+HALF.fais_mc);  

    if HALF.n_hc ~=0
        V_mc_kick  = V_mc_kick + HALF.rfcoef3*sin(HALF.rfcoef4*Q+HALF.fais_hc);
    end
    
    rad_quan_kick = -HALF.radampcoef * P + HALF.quanexcoef * ...
        gpuArray.randn(Par_num,Bun_num,'single');

    P = P + V_mc_kick + rad_quan_kick - HALF.ploss;
   
    % output data
    if mod(i,Recor_step)==0
        Q_line = reshape(Q,1,Total_num);
        P_line = reshape(P,1,Total_num);
        record_th = record_th +1;
        record_Q_mean(record_th)=gather(mean(Q_line));
        record_Q_std(record_th)=gather(std(Q_line));
        record_P_mean(record_th)=gather(mean(P_line));
        record_P_std(record_th)=gather(std(P_line));
    end
end

wait(gd);
toc;
tau_min = gather(Q_min)*sigma_t0;
bin_range = (1:binnum)*Dq*sigma_t0+tau_min;
filename=['MIGPU_n_',num2str(n_hc),'_I_tot_',num2str(I_tot*1e3),'mA.mat'];
save(filename,'record_Q_mean','record_Q_std','record_P_mean','record_P_std','Total_num','Dq','bin_range','bin_num_sumq','Recor_step','Recor_num','bin_num_sumq_gaussian','HALF','wake_kick');
end
end
%%
figure(1);
turns = (1:Recor_num)*Recor_step;
subplot(2,2,1)
plot(turns,record_Q_mean*HALF.sigma_t0*1e12); hold on;
subplot(2,2,2)
plot(turns,record_Q_std*HALF.sigma_t0*1e12); hold on;
subplot(2,2,3)
plot(turns,record_P_mean*HALF.sigma_e0); hold on;
subplot(2,2,4)
plot(turns,record_P_std*HALF.sigma_e0); hold on; 

subplot(2,2,1);ylabel('<\tau> [ps]');xlabel('turn');grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
subplot(2,2,2);ylabel('\sigma_{\tau} [ps]');xlabel('turn/');grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
subplot(2,2,3);ylabel('<\delta> ');xlabel('turn');grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
subplot(2,2,4);ylabel('\sigma_{\delta} ');xlabel('turn');grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
%%
% subplot(2,2,1);legend('binwidth 20E-4 ps','10E-4','5E-4','2E-4');
% subplot(2,2,2);legend('binwidth 20E-4 ps','10E-4','5E-4','2E-4');
% subplot(2,2,3);legend('binwidth 20E-4 ps','10E-4','5E-4','2E-4');
% subplot(2,2,4);legend('binwidth 20E-4 ps','10E-4','5E-4','2E-4');
%% 统计作密度分布
% tau_min = gather(Q_min)*sigma_t0;
% Dq = 0.01;
% Q_new = round((Q - Q_min) *(1/Dq));
% binnum=max(max(Q_new))+1; binnum=gather(binnum);
% bin_num_q=sum(BinNumCalZ(binnum,Q_new));
% bin_num_q=reshape(bin_num_q,binnum,Bun_num);
% bin_num_sumq = sum(bin_num_q,2);
% bin_range = (1:binnum)*Dq*sigma_t0+tau_min;
figure(2);
plot(bin_range'*1e12,bin_num_sumq/Total_num/(Dq*HALF.sigma_t0)/1e12,'.');hold on;
ylabel('Norm.density ');xlabel('\tau [ps]'); grid minor;
set(gca,'FontName','Times New Roman','FontSize',12);
%%
figure(3);
plot(bin_num_sumq_gaussian);hold on;
% legend('w HHC  @0.25 mA','w HHC  @0.5 mA','w HHC  @1.0 mA','w/o HHC @0.25 mA','w/o HHC @0.5 mA','w/o HHC @1.0 mA');
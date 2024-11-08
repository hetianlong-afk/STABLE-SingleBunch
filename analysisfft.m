% fft analysis
% FFT����Ƶɢ
% ����������Ƶ�ʼ����ɢ
% ��Ҫͳ��ÿȦ���ݣ�Ȧ����2��ָ��
clc;clear;
I_ave = [50:50:800,900:100:1600]*1e-3;
% I_ave = [245]*1e-3;
n_hc_scan = [0];        % г����    0 3
start = 3e3;
for ni = 1:length(n_hc_scan)
n_hc =  n_hc_scan(ni);
for Ii = 1:length(I_ave)
I0 = I_ave(Ii);

filename=['MIGPU_n_',num2str(n_hc),'_I_tot_',num2str(I0*1e3),'mA.mat'];
load(filename);
% mean_q = record_P_mean'; % ��P����Q, ��Q��Ҫ������ֱ������
mean_q = record_Q_mean(start:end)'-mean(record_Q_mean(start:end));
n_turns= length(mean_q);
% ͳ�����ĵ���Ƶ��
% ע��˴���ÿ10Ȧ��¼һ������
freqs = 0:1/n_turns:0.5;amp = abs(fft(mean_q));
plot(freqs/10*1/HALF.T0,amp(1:length(freqs))/max(amp(1:length(freqs)))); %/10 ��ʾÿ10Ȧ��¼һ������
xlim([0,1500]);hold on;
pause(0.1);
% hold on;
end
end
%%
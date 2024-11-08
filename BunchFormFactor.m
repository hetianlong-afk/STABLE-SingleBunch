function  BunchFormFactor(tau,lambda)
% 计算群聚因子
% tau 分布横坐标
% lambda 分布纵坐标

omega = [0.002:0.002:2]*1e12*2*pi;
omegaN = length(omega);
FW    = zeros(1,omegaN);
for i=1:omegaN
    FW(i)=abs(sum(lambda.*exp(1i*tau*omega(i))))^2;
end
fw = FW/max(FW);
figure(203)
semilogy(omega/2/pi/1e12,fw);
ylim([1e-8,1]);
xlabel('frequency [THz]');
ylabel('Bunching factor [a.u.]');
end
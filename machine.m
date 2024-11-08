function [HALF]=machine(C,I0,I_tot,U0,E0,tau_s,tau_z,sigma_t0,sigma_e0,alpha_c,h,...
    V_mc,n_hc,R_hc,Q_hc,fillrate,fre_shift,FPCondition)
% 创建机器参数，保存于HALF结构体中
cspeed = 299792458;
HALF.C       = C;
HALF.R       = C/(2*pi);
HALF.I0      = I0;
HALF.I_tot   = I_tot;
HALF.h       = h;
HALF.U0      = U0;
HALF.E0      = E0;
HALF.sigma_t0= sigma_t0;
HALF.sigma_e0= sigma_e0;
HALF.alpha_c = alpha_c;
HALF.tau_s   = tau_s;
HALF.tau_z   = tau_z;
HALF.V_mc    = V_mc;
HALF.n_hc    = n_hc;
HALF.R_hc    = R_hc;  
HALF.Q_hc    = Q_hc;
HALF.fillrate  = fillrate;
HALF.fre_shift = fre_shift;

HALF.T0 = HALF.C/cspeed;              % 回旋周期
HALF.Tb = HALF.T0/HALF.h;             % 相邻 norminal bucket position 距离
HALF.f_rf = HALF.h/HALF.T0;           % 高频频率
HALF.fre_hc = HALF.f_rf * HALF.n_hc;  % 谐波频率
HALF.w_rf = HALF.f_rf*2*pi;           % 高频角频率
HALF.wre_hc = HALF.fre_hc * 2*pi;     % 谐波角频率
HALF.qc = HALF.T0*HALF.I_tot/HALF.h/HALF.fillrate;    % bunch 电荷量
HALF.w_r = HALF.w_rf*HALF.n_hc+HALF.fre_shift*pi*2;   % 谐波谐振频率
HALF.angle = HALF.w_r * HALF.Tb;      % 相邻 norminal bucket center 旋转角
HALF.lambda_rads = HALF.T0 / HALF.tau_s;  % 辐射阻尼率   单位：圈数
HALF.lambda_radz = HALF.T0 / HALF.tau_z;  % 量子激发率   单位：圈数

HALF.fais_nat = pi-asin(HALF.U0/HALF.V_mc);% natural synchrotron phase
HALF.det_angle= atan(HALF.Q_hc*(HALF.w_r/HALF.wre_hc-HALF.wre_hc/HALF.w_r));
disp(['考虑谐波腔负载并假定均匀填充情况的失谐角度：',num2str(HALF.det_angle/pi*180),' deg']);
HALF.fais_mc_whc = pi - asin((HALF.U0+2*HALF.I0*HALF.R_hc*cos(HALF.det_angle)^2)/HALF.V_mc);
disp(['考虑谐波腔负载并假定均匀填充情况的同步相位：',num2str(HALF.fais_mc_whc),' rad']);
HALF.V_load_0 = 2*HALF.I0*HALF.R_hc*cos(HALF.det_angle)*exp(1i*(HALF.det_angle));
disp(['考虑谐波腔负载并假定均匀填充情况的初始负载：',num2str(HALF.V_load_0)]);

HALF.drift_coef = HALF.alpha_c * HALF.T0 * (HALF.sigma_e0 / HALF.sigma_t0);
HALF.kick_coef  = 1 / (HALF.E0 * HALF.sigma_e0);

if HALF.n_hc ~=0
    switch FPCondition
        case 0 % 依据实际负载腔压计算
            HALF.fais_mc = HALF.fais_mc_whc;
            HALF.fais_hc = HALF.det_angle-pi/2; 
            HALF.rfcoef3 = HALF.kick_coef * abs(HALF.V_load_0);
            HALF.rfcoef4 = HALF.w_rf * HALF.sigma_t0 * HALF.n_hc;
        case 1 % 最优拉伸的谐波腔腔压与相位
            n2       = HALF.n_hc^2;
            HALF.k_fp     = sqrt(1/n2-1/(n2-1)*(HALF.U0/HALF.V_mc)^2); % HC 腔压系数
            HALF.fais_fp  = pi-asin(n2/(n2-1)*HALF.U0/HALF.V_mc);      % MC 同步相位
            HALF.fais_mc  = HALF.fais_fp;
            HALF.nfaih_fp = atan(tan(HALF.fais_fp)/HALF.n_hc);         % HC 同步相位
            HALF.fais_hc  = HALF.nfaih_fp;
            HALF.rfcoef3 = HALF.kick_coef * HALF.V_mc * HALF.k_fp;
            HALF.rfcoef4 = HALF.w_rf * HALF.sigma_t0 * HALF.n_hc;
    end
else
    HALF.fais_mc = HALF.fais_nat;
end

HALF.rfcoef1 = HALF.kick_coef * HALF.V_mc;
HALF.rfcoef2 = HALF.w_rf * HALF.sigma_t0;

HALF.radampcoef = 2 * HALF.lambda_rads;
HALF.quanexcoef = 2 * sqrt(HALF.lambda_radz);

HALF.ploss = HALF.kick_coef * HALF.U0;
end
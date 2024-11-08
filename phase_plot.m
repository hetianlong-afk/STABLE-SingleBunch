function phase_plot(Q,P,Bun_num,Par_num,sigma_t0,sigma_e0,i,sigr)
% Q - GPUARRAY 2D
% P - GPUARRAY 2D
% phase_plot(Q,P,Bun_num,Par_num,sigma_t0,sigma_e0,sigr)
x = gather(reshape(Q,1,Bun_num*Par_num));
y = gather(reshape(P,1,Bun_num*Par_num));

% [Intensity,Xedges,Yedges] = histcounts2(x,y,500,'Normalization','probability');
[Intensity,Xedges,Yedges] = histcounts2(y,x,500,'Normalization','probability');

% [xx,yy]=meshgrid(Xedges(2:end),Yedges(2:end));
% qline=-sigr:0.05:sigr;
% pline=-sigr:0.05:sigr;
% [qq,pp]=meshgrid(qline,pline);
% NIntensity = interp2(xx,yy,Intensity,qq,pp,'bicubic');
% plot3D(qq*sigma_t0*1e12,pp*sigma_e0,NIntensity,20);
% imagesc(qline*sigma_t0*1e12,qline*sigma_e0,flipud(NIntensity));
imagesc(Xedges*sigma_t0*1e12,Yedges*sigma_e0,Intensity);axis xy;
% imagesc(Xedges,Yedges,Intensity);
shading interp; 
colorbar;
colormap(hot);
xlim([min(Xedges)*sigma_t0*1e12,max(Xedges)*sigma_t0*1e12]);
ylim([min(Yedges)*sigma_e0,max(Yedges)*sigma_e0]);
xlabel('\tau [ps]');ylabel('\delta');
title(['Turn=',num2str(i)]);
% figure(201);
% plot(sum(Intensity,1));hold on;
% plot(x1);
% figure(202);
% plot(sum(Intensity,2));hold on;
% plot(y1);

% x = gather(reshape(Q,1,Bun_num*Par_num));
% y = gather(reshape(P,1,Bun_num*Par_num));
% [Intensity,Xedges,Yedges] = histcounts2(y,x,500,'Normalization','probability');
% 
% [xx,yy]=meshgrid(Xedges(2:end),Yedges(2:end));
% qline=-sigr:0.05:sigr;
% pline=-sigr:0.05:sigr;
% [qq,pp]=meshgrid(qline,pline);
% NIntensity = interp2(xx,yy,Intensity,qq,pp,'bicubic');
% 
% set(gcf,'Color',[1,1,1]);
% % ax1
% ax1=axes('Parent',gcf);
% hold(ax1,'on');
% imagesc(qline,qline,NIntensity);
% colormap(hot);
% % contour(qq,pp,NIntensity,20);
% hold(ax1,'off');
% xlabel('q');
% ylabel('p');
% xlim([-sigr,sigr]);
% ylim([-sigr,sigr]);
% ax1.XColor='none';
% ax1.YColor='none';
% % xlabel('\tau/\sigma_{\tau}');ylabel('\delta/\sigma_{\delta}');
% % shading interp; 
% % colormap(jet);
% ax1.Position=[0.1,0.12,0.6,0.6];
% % ax1.TickDir='out';
% % ax1.XTickLabel='';
% % ax1.YTickLabel='';
% % ax2
% ax2=axes('Parent',gcf);
% hold(ax2,'on');
% plot(Xedges(2:end),sum(Intensity),'r'); 
% hold(ax2,'off');
% ax2.Position=[0.12,0.75,0.6,0.20];
% ax2.XColor='none';
% ax2.YColor='none';
% ax2.YTickLabel='';
% ax2.XLim=ax1.XLim;
% % ax3
% ax3=axes('Parent',gcf);
% hold(ax3,'on');
% plot(sum(Intensity,2),Yedges(2:end),'b'); 
% hold(ax3,'off');
% ax3.Position=[0.75,0.12,0.20,0.6];
% ax3.XColor='none';
% ax3.YColor='none';
% ax3.YTickLabel='';
% ax3.YLim=ax1.YLim;

BunchFormFactor(Xedges(2:end)*sigma_t0,sum(Intensity));
end
clear all
clc
load SANEPAR_COMPLETO.txt

est = 550:1150;
u1e = SANEPAR_COMPLETO(est,1)';
u2e = SANEPAR_COMPLETO(est,2)';
u3e = SANEPAR_COMPLETO(est,3)';
u4e = SANEPAR_COMPLETO(est,4)';
Ue = [u1e;u2e;u3e;u4e];
y1e = SANEPAR_COMPLETO(est,5)';
y2e = SANEPAR_COMPLETO(est,6)';
y3e = SANEPAR_COMPLETO(est,7)';
Ye = [y1e;y2e;y3e];

val = 1500:2000;
u1v = SANEPAR_COMPLETO(val,1)';
u2v = SANEPAR_COMPLETO(val,2)';
u3v = SANEPAR_COMPLETO(val,3)';
u4v = SANEPAR_COMPLETO(val,4)';
Uv = [u1v;u2v;u3v;u4v];
y1v = SANEPAR_COMPLETO(val,5)';
y2v = SANEPAR_COMPLETO(val,6)';
y3v = SANEPAR_COMPLETO(val,7)';
Yv = [y1v;y2v;y3v];

load('NSWRSO.mat')
    PF = [ ];
    PS = [ ];
for k = 1:50
    A = PARETO_FRONT(:,:,k);
    B = PARETO_SET(:,:,k);
    PF = [PF A];
    PS = [PS B];
end
[MSEG_est,MSE_est,R2G_est,R2_est] = fun_objetivo_gera_res(PS,Ue,Ye);
[MSEG_val,MSE_val,R2G_val,R2_val] = fun_objetivo_gera_res(PS,Uv,Yv);
[linha,coluna] = size(R2_val);
dis = sqrt((sum(ones(3,10000)- R2_val).^2));
[minimo,pos] = min(dis);
xbest_NSWRSO = PS(:,pos);
load('NSGA2.mat')
    PF = [ ];
    PS = [ ];
for k = 1:50
    A = PARETO_FRONT(:,:,k);
    B = PARETO_SET(:,:,k);
    PF = [PF A];
    PS = [PS B];
end
[MSEG_est,MSE_est,R2G_est,R2_est] = fun_objetivo_gera_res(PS,Ue,Ye);
[MSEG_val,MSE_val,R2G_val,R2_val] = fun_objetivo_gera_res(PS,Uv,Yv);
[linha,coluna] = size(R2_val);
dis = sqrt((sum(ones(3,10000)- R2_val).^2));
[minimo,pos] = min(dis);
xbest_NSGA2 = PS(:,pos);
load('MOGWO.mat')
    PF = [ ];
    PS = [ ];
for k = 1:50
    A = PARETO_FRONT(:,:,k);
    B = PARETO_SET(:,:,k);
    PF = [PF A];
    PS = [PS B];
end
[MSEG_est,MSE_est,R2G_est,R2_est] = fun_objetivo_gera_res(PS,Ue,Ye);
[MSEG_val,MSE_val,R2G_val,R2_val] = fun_objetivo_gera_res(PS,Uv,Yv);
[linha,coluna] = size(R2_val);
dis = sqrt((sum(ones(3,10000)- R2_val).^2));
[minimo,pos] = min(dis);
xbest_MOGWO = PS(:,pos);

%estimação
figure(1)
[Y_solucoes_est1] = fun_objetivo_gera_curva(xbest_NSWRSO,est);
[Y_solucoes_est2] = fun_objetivo_gera_curva(xbest_NSGA2,est);
[Y_solucoes_est3] = fun_objetivo_gera_curva(xbest_MOGWO,est);
ye_NSWRSO=Y_solucoes_est1{1,1};
ye_NSGA2=Y_solucoes_est2{1,1};
ye_MOGWO=Y_solucoes_est3{1,1};
figure(1)
subplot 311
plot(Ye(1,:),'k')
%title('Estimation')
xlabel('Time (hours)')
xlim([0 600])
ylabel('Water flow (l/s)')
hold on
plot(ye_NSWRSO(:,1),'b')
plot(ye_NSGA2(:,1),'g')
plot(ye_MOGWO(:,1),'r')
subplot 312
plot(Ye(2,:),'k')
xlabel('Time (hours)')
ylabel('Level (m)')
xlim([0 600])
hold on
plot(ye_NSWRSO(:,2),'b')
plot(ye_NSGA2(:,2),'g')
plot(ye_MOGWO(:,2),'r')
subplot 313
plot(Ye(3,:),'k')
xlabel('Time (hours)')
ylabel('Pressure (mca)')
xlim([0 600])
hold on
plot(ye_NSWRSO(:,3),'b')
plot(ye_NSGA2(:,3),'g')
plot(ye_MOGWO(:,3),'r')

%estimação
figure(1)
[Y_solucoes_est1] = fun_objetivo_gera_curva(xbest_NSWRSO,est);
[Y_solucoes_est2] = fun_objetivo_gera_curva(xbest_NSGA2,est);
[Y_solucoes_est3] = fun_objetivo_gera_curva(xbest_MOGWO,est);
ye_NSWRSO=Y_solucoes_est1{1,1};
ye_NSGA2=Y_solucoes_est2{1,1};
ye_MOGWO=Y_solucoes_est3{1,1};
figure(1)
subplot 311
plot(Ye(1,:),'k')
xlabel('Time (hours)')
xlim([0 600])
ylabel('Water flow (l/s)')
hold on
plot(ye_NSWRSO(:,1),'b')
plot(ye_NSGA2(:,1),'g')
plot(ye_MOGWO(:,1),'r')
subplot 312
plot(Ye(2,:),'k')
xlabel('Time (hours)')
ylabel('Level (m)')
xlim([0 600])
hold on
plot(ye_NSWRSO(:,2),'b')
plot(ye_NSGA2(:,2),'g')
plot(ye_MOGWO(:,2),'r')
subplot 313
plot(Ye(3,:),'k')
xlabel('Time (hours)')
ylabel('Pressure (mca)')
xlim([0 600])
hold on
plot(ye_NSWRSO(:,3),'b')
plot(ye_NSGA2(:,3),'g')
plot(ye_MOGWO(:,3),'r')

%estimação
[Y_solucoes_est1] = fun_objetivo_gera_curva(xbest_NSWRSO,est);
[Y_solucoes_est2] = fun_objetivo_gera_curva(xbest_NSGA2,est);
[Y_solucoes_est3] = fun_objetivo_gera_curva(xbest_MOGWO,est);
ye_NSWRSO=Y_solucoes_est1{1,1};
ye_NSGA2=Y_solucoes_est2{1,1};
ye_MOGWO=Y_solucoes_est3{1,1};
figure(1)
subplot 311
plot(Ye(1,:),'k')
title('Estimation')
xlabel('Time (hours)')
xlim([0 600])
ylabel('Water flow (l/s)')
hold on
plot(ye_NSWRSO(:,1),'b')
plot(ye_NSGA2(:,1),'g')
plot(ye_MOGWO(:,1),'r')
subplot 312
plot(Ye(2,:),'k')
xlabel('Time (hours)')
ylabel('Level (m)')
xlim([0 600])
hold on
plot(ye_NSWRSO(:,2),'b')
plot(ye_NSGA2(:,2),'g')
plot(ye_MOGWO(:,2),'r')
subplot 313
plot(Ye(3,:),'k')
xlabel('Time (hours)')
ylabel('Pressure (mca)')
xlim([0 600])
hold on
plot(ye_NSWRSO(:,3),'b')
plot(ye_NSGA2(:,3),'g')
plot(ye_MOGWO(:,3),'r')

%validação
figure(2)
[Y_solucoes_val1] = fun_objetivo_gera_curva(xbest_NSWRSO,val);
[Y_solucoes_val2] = fun_objetivo_gera_curva(xbest_NSGA2,val);
[Y_solucoes_val3] = fun_objetivo_gera_curva(xbest_MOGWO,val);
yv_NSWRSO=Y_solucoes_val1{1,1};
yv_NSGA2=Y_solucoes_val2{1,1};
yv_MOGWO=Y_solucoes_val3{1,1};
subplot 311
plot(Yv(1,:),'k')
title('Validation')
xlabel('Time (hours)')
xlim([0 500])
ylabel('Water flow (l/s)')
hold on
plot(yv_NSWRSO(:,1),'b')
plot(yv_NSGA2(:,1),'g')
plot(yv_MOGWO(:,1),'r')
subplot 312
plot(Yv(2,:),'k')
xlabel('Time (hours)')
ylabel('Level (m)')
xlim([0 500])
hold on
plot(yv_NSWRSO(:,2),'b')
plot(yv_NSGA2(:,2),'g')
plot(yv_MOGWO(:,2),'r')
subplot 313
plot(Yv(3,:),'k')
xlabel('Time (hours)')
ylabel('Pressure (mca)')
xlim([0 500])
hold on
plot(yv_NSWRSO(:,3),'b')
plot(yv_NSGA2(:,3),'g')
plot(yv_MOGWO(:,3),'r')
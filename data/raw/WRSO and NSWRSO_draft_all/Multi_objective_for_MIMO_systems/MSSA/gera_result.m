clear all
clc
load SANEPAR_COMPLETO.txt
% Estimação 1
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

load('Result Table2.mat')
PF = [];
PS = []; 

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
xbest = PS(:,pos);
MSE_est_best = MSE_est(:,pos);
MSE_val_best = MSE_val(:,pos);
MSEG_est_best = MSEG_est(1,pos);
MSEG_val_best = MSEG_val(1,pos);
R2_est_best = R2_est(:,pos);
R2_val_best = R2_val(:,pos);
R2G_est_best = R2G_est(1,pos);
R2G_val_best = R2G_val(1,pos);

%estimação
figure(1)
[Y_solucoes_est] = fun_objetivo_gera_curva(xbest,est);
ye=Y_solucoes_est{1,1};
figure(1)
%Validação
subplot 311
plot(Ye(1,:))
title('Estimação')
xlabel('Tempo (horas)')
ylabel('Vazão (m³/s)')
hold on
plot(ye(:,1))
subplot 312
plot(Ye(2,:))
xlabel('Tempo (horas)')
ylabel('Nível (metros)')
hold on
plot(ye(:,2))
subplot 313
plot(Ye(3,:))
xlabel('Tempo (horas)')
ylabel('Pressão (mca)')
hold on
plot(ye(:,3))
%validação
figure(2)
[Y_solucoes_val] = fun_objetivo_gera_curva(xbest,val);
yv=Y_solucoes_val{1,1};
figure(2)
%Validação
subplot 311
plot(Yv(1,:))
title('Validação')
xlabel('Tempo (horas)')
ylabel('Vazão (m³/s)')
hold on
plot(yv(:,1))
subplot 312
plot(Yv(2,:))
xlabel('Tempo (horas)')
ylabel('Nível (metros)')
hold on
plot(yv(:,2))
subplot 313
plot(Yv(3,:))
xlabel('Tempo (horas)')
ylabel('Pressão (mca)')
hold on
plot(yv(:,3))
load results.mat

x = Archive(1).Position';
[SSEG,Yp] = fun_curvas(x,Ue,Ye);
Y_est = Yp';
figure(1)
%Estimação
subplot 311
plot(Ye(1,:))
title('Estimação')
xlabel('Tempo (horas)')
ylabel('Vazão (m³/s)')
hold on
plot(Y_est(1,:))
subplot 312
plot(Ye(2,:))
xlabel('Tempo (horas)')
ylabel('Nível (metros)')
hold on
plot(Y_est(2,:))
subplot 313
plot(Ye(3,:))
xlabel('Tempo (horas)')
ylabel('Pressão (mca)')
hold on
plot(Y_est(3,:))

est = 1500:2000;
u1v = SANEPAR_COMPLETO(est,1)';
u2v = SANEPAR_COMPLETO(est,2)';
u3v = SANEPAR_COMPLETO(est,3)';
u4v = SANEPAR_COMPLETO(est,4)';
Uv = [u1v;u2v;u3v;u4v];
y1v = SANEPAR_COMPLETO(est,5)';
y2v = SANEPAR_COMPLETO(est,6)';
y3v = SANEPAR_COMPLETO(est,7)';
Yv = [y1v;y2v;y3v];
[SSEG,Yp] = fun_curvas(x,Uv,Yv);
Y_val = Yp';
figure(2)
%Estimação
subplot 311
plot(Yv(1,:))
title('Estimação')
xlabel('Tempo (horas)')
ylabel('Vazão (m³/s)')
hold on
plot(Y_val(1,:))
subplot 312
plot(Yv(2,:))
xlabel('Tempo (horas)')
ylabel('Nível (metros)')
hold on
plot(Y_val(2,:))
subplot 313
plot(Yv(3,:))
xlabel('Tempo (horas)')
ylabel('Pressão (mca)')
hold on
plot(Y_val(3,:))
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
Max_Gen =1000;%8000
runtimes=50;  
Hypervolume = zeros(1,runtimes);
search_agents = 200;
dimension = 72;
n_obj = 3;
R2=zeros(1,runtimes);
xbest=zeros(dimension,runtimes);
tempo=zeros(1,runtimes);
Archive_size=200;
PARETO_FRONT=zeros(3,Archive_size,runtimes);
PARETO_SET=zeros(72,Archive_size,runtimes);
for k=1:runtimes
tic
[Archive_X,Archive_F] = MSSA(Ue,Ye,n_obj,search_agents,Max_Gen,dimension);
tempo(1,k)= toc;

[comp,num_sol_n_dominadas]=size(Archive_F);

for j = 1:num_sol_n_dominadas
PARETO_FRONT(:,j,k) = Archive_F(:,j);
PARETO_SET(:,j,k) = Archive_X(:,j);
end
k
txt = 'Result Table2';
save (txt, 'PARETO_FRONT','PARETO_SET','tempo');
end
txt = 'Result Table2';
save (txt, 'PARETO_FRONT','PARETO_SET','tempo');
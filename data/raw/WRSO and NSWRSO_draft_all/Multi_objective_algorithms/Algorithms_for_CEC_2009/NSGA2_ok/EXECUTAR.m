clear all 
clc
nn = 1;
IGD = zeros(1,10);
obj_no=2;
CUSTO_TOTAL = zeros(obj_no,100,10);
dim=10;
TestProblem='UF8';
while(nn <= 1)
 %  MRSO
   [Archive_X,Archive_F] = NSGAII(dim,obj_no,TestProblem);
[num_sol_n_dominadas,comp]=size(Archive_X);

Custo = zeros(obj_no,num_sol_n_dominadas);
for i = 1:num_sol_n_dominadas
Custo(:,i) = Archive_F(:,i);
end
CUSTO_TOTAL(:,1:num_sol_n_dominadas,nn) = Custo;%
%custo_min = 1 - (Custo(1,:)).^2-Custo(2,:);%f4
custo_min = 1 - Custo(1,:) - Custo(2,:); %f5 f6 f7
%custo_min = 1 - Custo(1,:) - Custo(2,:) - Custo(3,:);%f9
%custo_min = 1 - Custo(1,:).^2 - (Custo(2,:).^2) - (Custo(3,:).^3); %f10 f8
%custo_min = 1 - sqrt(Custo(1,:))-Custo(2,:);%f1 f2 f3
%[c_best,n_best] = min(sum(Custo));
[c_best,n_best] = min(abs(custo_min));

d=zeros(1,num_sol_n_dominadas);
for i = 1:num_sol_n_dominadas
d(1,i) = sqrt(sum((Custo(:,i) - Custo(:,n_best)).^2));
end

IGD(1,nn) = sqrt(sum(d.^2))/num_sol_n_dominadas;
nn=nn+1

end
save results
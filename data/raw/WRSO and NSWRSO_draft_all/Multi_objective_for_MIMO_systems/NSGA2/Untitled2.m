clear all  
   PARETO_FRONTd = zeros(3,200,100);
   PARETO_SETd = zeros(72,200,100);
   tempod=zeros(1,100);

load('Result Table.mat')
for i =1:51
   PARETO_FRONTd(:,:,i) = PARETO_FRONT(:,:,i);
   PARETO_SETd(:,:,i)=PARETO_SET(:,:,i);
   tempod(:,i)=tempo(:,i);
end
load('Result Table2.mat')
for i =52:100
   PARETO_FRONTd(:,:,i) = PARETO_FRONT(:,:,i);
   PARETO_SETd(:,:,i)=PARETO_SET(:,:,i);
   tempod(:,i)=tempo(:,i);
end
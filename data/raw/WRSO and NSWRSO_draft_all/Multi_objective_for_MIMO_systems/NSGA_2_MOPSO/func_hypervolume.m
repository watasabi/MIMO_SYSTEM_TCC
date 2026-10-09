repoint_2D = [1,1];
repoint_3D = [1,1,1];
HP = zeros(1,10);
load('UF_1.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_2D);
end
HP(1,1) = mean(hp);
load('UF_2.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_2D);
end
HP(1,2) = mean(hp);
load('UF_3.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_2D);
end
HP(1,3) = mean(hp);
load('UF_4.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_2D);
end
HP(1,4) = mean(hp);
load('UF_5.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_2D);
end
HP(1,5) = mean(hp);
load('UF_6.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_2D);
end
HP(1,6) = mean(hp);
load('UF_7.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_2D);
end
HP(1,7) = mean(hp);
load('UF_8.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_3D);
end
HP(1,8) = mean(hp);
load('UF_9.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_3D);
end
HP(1,9) = mean(hp);
load('UF_10.mat')
for i=1:10
pf=CUSTO_TOTAL(:,:,i)';
hp(i) = Hypervolume_calculation(pf,repoint_3D);
end
HP(1,10) = mean(hp);
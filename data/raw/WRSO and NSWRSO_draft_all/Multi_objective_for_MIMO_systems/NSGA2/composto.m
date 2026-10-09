   R2d = zeros(1,100);
   tempod=zeros(1,100);
   xbestd=zeros(72,100);

load('matlab.mat')
for i =1:70
   R2d(1,i) = R2(1,i);
   tempod(1,i)=tempo(1,i);
   xbestd(:,i)=xbest(:,i);
end
load('result1.mat')
for i =1:30
   R2d(1,i+70) = R2(1,i);
   tempod(1,i+70)=tempo(1,i);
   xbestd(:,i+70)=xbest(:,i);
end
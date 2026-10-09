function [FoodPosition,FoodFitness,Convergence_best,Convergence_mean]= SSA_func(pop_initial,fhd,Dimension,Particle_Number,Max_Gen,VRmin,VRmax,varargin)
rand('state',sum(100*clock));
me=Max_Gen;
ps=Particle_Number;
D=Dimension;
FoodPosition=zeros(1,D);
FoodFitness=inf;
Convergence_best = zeros(1,me);
Convergence_mean = zeros(1,me);
t=0;
if length(VRmin)==1
    VRmin=repmat(VRmin,1,D);
    VRmax=repmat(VRmax,1,D);
end
mv=0.5*(VRmax-VRmin);
VRmin=repmat(VRmin,ps,1);
VRmax=repmat(VRmax,ps,1);

SalpPositions=pop_initial;
for i=1:size(SalpPositions,1)
    SalpFitness(1,i)=feval(fhd,SalpPositions(i,:)',varargin{:});
end
[sorted_salps_fitness,sorted_indexes]=sort(SalpFitness);
for newindex=1:ps
    Sorted_salps(newindex,:)=SalpPositions(sorted_indexes(newindex),:);
end
FoodPosition=Sorted_salps(1,:);
FoodFitness=sorted_salps_fitness(1);
while t<me
       c1 = 2*exp(-(4*t/me)^2); % Eq. (3.2) in the paper

    for i=1:size(SalpPositions,1)
        
        SalpPositions= SalpPositions';
        
        if i<=ps/2
            for j=1:1:D
                c2=rand();
                c3=rand();
                %%%%%%%%%%%%% % Eq. (3.1) in the paper %%%%%%%%%%%%%%
                if c3<0.5 
                    SalpPositions(j,i)=FoodPosition(j)+c1*((VRmax(j)-VRmin(j))*c2+VRmin(j));
                else
                    SalpPositions(j,i)=FoodPosition(j)-c1*((VRmax(j)-VRmin(j))*c2+VRmin(j));
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
        elseif i>ps/2 && i<ps+1
            point1=SalpPositions(:,i-1);
            point2=SalpPositions(:,i);
            SalpPositions(:,i)=(point2+point1)/2; 
          %  SalpPositions(:,i)=(SalpFitness(1,i-1)*point2+SalpFitness(1,i)*point1)/(SalpFitness(1,i-1)+SalpFitness(1,i)); % % Eq. (3.4) in the paperSalpFitness(1,i)
        end
        
        SalpPositions= SalpPositions';
    end
    
   for i=1:size(SalpPositions,1)
        
        Tp=SalpPositions(i,:)>VRmax(i,:);Tm=SalpPositions(i,:)<VRmin(i,:);SalpPositions(i,:)=(SalpPositions(i,:).*(~(Tp+Tm)))+VRmax(i,:).*Tp+VRmin(i,:).*Tm;
        FoodFitnessb=max(SalpFitness);
        SalpFitness(1,i)=feval(fhd,SalpPositions(i,:)',varargin{:});
        FoodFitnessa=min(SalpFitness);
        if SalpFitness(1,i)<FoodFitness
            FoodPosition=SalpPositions(i,:);
            FoodFitness=SalpFitness(1,i);
            
        end
    end
    t=t+1;
    Convergence_best(t)=FoodFitness;
    Convergence_mean(t)=mean(SalpFitness);
end

end




function [Position,Leader_score,Convergence_best,Convergence_mean]= RSO_func(pop_initial,fhd,Dimension,Particle_Number,Max_Gen,VRmin,VRmax,varargin)
rand('state',sum(100*clock));
me=Max_Gen;
ps=Particle_Number;
D=Dimension;
Position=zeros(1,D);
Leader_score=inf; 

l=0;
x = 1;
y = 5;
R = floor((y-x).*rand(1,1) + x);
Convergence_best = zeros(1,me);
Convergence_mean = zeros(1,me);
fitness = zeros(1,ps);
if length(VRmin)==1
    VRmin=repmat(VRmin,1,D);
    VRmax=repmat(VRmax,1,D);
end
mv=0.5*(VRmax-VRmin);
VRmin=repmat(VRmin,ps,1);
VRmax=repmat(VRmax,ps,1);

Positions=pop_initial;


while l<me
    for i=1:size(Positions,1)  

        fitness(i) = feval(fhd,Positions(i,:)',varargin{:});
        if fitness(i)<Leader_score 
            Leader_score=fitness(i); 
            Position=Positions(i,:);
        end
        
    end
   
    A=R-l*((R)/me); 
    
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
            C=2*rand();          
            P_vec=A*Positions(i,j)+abs(C*((Position(j)-Positions(i,j))));                   
            P_final=Position(j)-P_vec;
            Positions(i,j)=P_final;
        end
    end
     Tp=Positions(i,:)>VRmax(i,:);Tm=Positions(i,:)<VRmin(i,:);Positions(i,:)=(Positions(i,:).*(~(Tp+Tm)))+VRmax(i,:).*Tp+VRmin(i,:).*Tm;
   
    l=l+1;    
    Convergence_best(l)=Leader_score;
    Convergence_mean(l)=mean(fitness);
end

end




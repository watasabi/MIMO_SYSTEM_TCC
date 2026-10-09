function [Leader_pos,Leader_score,Convergence_best,Convergence_mean]= WRSO_func(pop_initial,fhd,Dimension,Particle_Number,Max_Gen,VRmin,VRmax,varargin)
rand('state',sum(100*clock));
me=Max_Gen;
ps=Particle_Number;
D=Dimension;
Leader_pos=zeros(1,D);
Leader_score=inf; 

t=0;
x = 1;
y = 5;

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


while t<me
     R = floor((y-x).*rand(Particle_Number,D) + x);
    for i=1:size(Positions,1)  

        fitness(i) = feval(fhd,Positions(i,:)',varargin{:});
        if fitness(i)<Leader_score 
            Leader_score=fitness(i); 
            Leader_pos=Positions(i,:);
        end
        
    end
    
    a=2-t*((2)/me); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+t*((-1)/me);
    
    % Update the Position of search agents 
    r1 = rand(Particle_Number,1);
    r2 = rand(Particle_Number,1);
    A=2*a*r1-a;
    C=2*r2;
    b=1;               %  parameters in Eq. (2.5)
    l=(a2-1)*rand(Particle_Number,1)+1;   %  parameters in Eq. (2.5)
    p = rand(Particle_Number,1);       % p in Eq. (2.6)
    
    rand_leader_index = floor(Particle_Number*rand(Particle_Number,1)+1);
    X_rand = Positions(rand_leader_index, :);
    D_X_rand=abs((C*ones(1,D)).*X_rand - Positions); % Eq. (2.7)
    Positions1 = X_rand - (A*ones(1,D)).*D_X_rand;     
    
    Ar = R - t*(R/me); 
    Cr = 2*rand(Particle_Number,D);
    P_vec=Ar.*Positions + abs(Cr.*(ones(Particle_Number,1)*Leader_pos-Positions));                   
    Positions2=ones(Particle_Number,1)*Leader_pos-P_vec;
    
    distance2Leader=abs(ones(Particle_Number,1)*Leader_pos-Positions);
    Positions3 = distance2Leader.*exp(b.*l).*cos(l.*2*pi)+ones(Particle_Number,1)*Leader_pos;
    
    
    Positions(p<0.5 & abs(A)>=1,:) = Positions1(p<0.5 & abs(A)>=1,:);            
    Positions(p<0.5 & abs(A)<1,:) = Positions2(p<0.5 & abs(A)<1,:);  
    Positions(p>=0.5,:) = Positions3(p>=0.5,:); 

    Tp=Positions(i,:)>VRmax(i,:);Tm=Positions(i,:)<VRmin(i,:);Positions(i,:)=(Positions(i,:).*(~(Tp+Tm)))+VRmax(i,:).*Tp+VRmin(i,:).*Tm;
    
    
    t=t+1; 
    Convergence_best(t)=Leader_score;
    Convergence_mean(t)=mean(fitness);
end

end




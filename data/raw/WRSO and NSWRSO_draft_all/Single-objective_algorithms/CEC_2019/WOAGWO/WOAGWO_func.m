function [Alpha_pos,Alpha_score,Convergence_best,Convergence_mean]= WOAGWO_func(pop_initial,fhd,Dimension,Particle_Number,Max_Gen,VRmin,VRmax,varargin)
rand('state',sum(100*clock));
me=Max_Gen;
ps=Particle_Number;
D=Dimension;
% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,D);
Alpha_score=inf; %change this to -inf for maximization problems
Beta_pos=zeros(1,D);
Beta_score=inf; %change this to -inf for maximization problems
Delta_pos=zeros(1,D);
Delta_score=inf; %change this to -inf for maximization problems
z=zeros(ps,D);
Convergence_best = zeros(1,me);
Convergence_mean = zeros(1,me);
fitness = zeros(1,ps);
t=0;
if length(VRmin)==1
    VRmin=repmat(VRmin,1,D);
    VRmax=repmat(VRmax,1,D);
end
VRmin=repmat(VRmin,ps,1);
VRmax=repmat(VRmax,ps,1);
Positions=pop_initial;
while t<me
    for i=1:size(Positions,1)  
        fitness(i) = feval(fhd,Positions(i,:)',varargin{:});
        % Update Alpha, Beta, and Delta
        if fitness(i)<Alpha_score 
            Alpha_score=fitness(i); % Update alpha
            Alpha_pos=Positions(i,:);
        end
        if fitness(i)>Alpha_score && fitness(i)<Beta_score 
            Beta_score=fitness(i); % Update beta
            Beta_pos=Positions(i,:);
        end
        if fitness(i)>Alpha_score && fitness(i)>Beta_score && fitness(i)<Delta_score 
            Delta_score=fitness(i); % Update delta
            Delta_pos=Positions(i,:);
        end
    end
    a=2-t*((2)/me); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+t*((-1)/me);
    % Update the Position of search agents 
    for i=1:size(Positions,1)
       r1=rand(); % r1 is a random number in [0,1]
       r2=rand(); % r2 is a random number in [0,1]
       A=2*a*r1-a;  % Eq. (2.3) in the paper
       C=2*r2;      % Eq. (2.4) in the paper
       b=1;               %  parameters in Eq. (2.5)
       l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)   
       p = rand();        % p in Eq. (2.6)
        for j=1:size(Positions,2)
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            r1=rand();
            r2=rand();  
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            r1=rand();
            r2=rand(); 
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
          if p<0.5              
               if abs(A)>=1
               rand_leader_index = floor(ps*rand()+1);
               X_rand = Positions(rand_leader_index, :);
               D_X_rand=abs(C*X_rand(j)-Positions(i,j)); 
               z(i,:)=X_rand(j)-A*D_X_rand;
               % Evaluate new solutions
               Fnew = feval(fhd,z(i,:)',varargin{:});
%              Update if the solution improves, or not too l
               if (Fnew<=fitness(i)) 
               Positions(i,:)=z(i,:);
               fitness(i)=Fnew;
               end      
               elseif abs(A)<1
                   D_Leader=abs(C*Alpha_pos(j)-Positions(i,j)); 
                   z(i,:)=Alpha_pos(j)-A*D_Leader;
    %              Evaluate new solutions
                   Fnew = feval(fhd,z(i,:)',varargin{:});
    %              Update if the solution improves, or not too loud
                   if (Fnew<=fitness(i)) 
                   Positions(i,:)=z(i,:);
                   fitness(i)=Fnew;
                   end
               end           
           elseif p>=0.5
               % Exploitation Phase   
               if((A1>-1 || A1<1)&&(A2>-1 || A2<1)&&(A3>-1 || A3<1))            
                D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
                X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
                X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
                D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
                X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
                Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
               end
           end     
        end
        Tp=Positions(i,:)>VRmax(i,:);Tm=Positions(i,:)<VRmin(i,:);Positions(i,:)=(Positions(i,:).*(~(Tp+Tm)))+VRmax(i,:).*Tp+VRmin(i,:).*Tm;    
    end 
    t=t+1; 
    Convergence_best(t)=Alpha_score;
    Convergence_mean(t)=mean(fitness);
end
end




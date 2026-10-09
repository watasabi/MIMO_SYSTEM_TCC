function [Leader_pos,Leader_score,Convergence_best,Convergence_mean]= WOA_func(pop_initial,fhd,Dimension,Particle_Number,Max_Gen,VRmin,VRmax,varargin)
rand('state',sum(100*clock));
me=Max_Gen;
ps=Particle_Number;
D=Dimension;
Leader_pos=zeros(1,D);
Leader_score=inf; 
Convergence_best = zeros(1,me);
Convergence_mean = zeros(1,me);
fitness = zeros(1,ps);
%bat algorithm addition
Qmin=0;         % Frequency minimum
Qmax=2;         % Frequency maximum
Q=zeros(ps,1);   % Frequency
v=zeros(ps,D);   % Velocities
z=zeros(ps,D);   
r=0.5;
A1=0.5;

if length(VRmin)==1
    VRmin=repmat(VRmin,1,D);
    VRmax=repmat(VRmax,1,D);
end
mv=0.5*(VRmax-VRmin);
VRmin=repmat(VRmin,ps,1);
VRmax=repmat(VRmax,ps,1);
Positions=pop_initial;

t=0;
while t<me
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
    for i=1:size(Positions,1)
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper
        
        
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
        
        p = rand();        % p in Eq. (2.6)
        
        for j=1:size(Positions,2)
            
            if p<0.5   
                
                if abs(A)>=1

                             rand_leader_index = floor(ps*rand()+1);
                             X_rand = Positions(rand_leader_index, :);
                             Q(i)=Qmin+(Qmin-Qmax)*rand;
                             v(i,:)=v(i,j)+(X_rand(j)-Leader_pos(j))*Q(i);
                            z(i,:)= Positions(i,:)+ v(i,:);
                            
                            
                            %%%% problem 
                                if rand>r
                                % The factor 0.001 limits the step sizes of random walks 
                                z(i,:)=Leader_pos(j)+0.001*randn(1,D);
                                end
                                     % Evaluate new solutions
                                    Fnew=feval(fhd,z(i,:)',varargin{:});
                                     % Update if the solution improves, or not too loud
                                    if (Fnew<=fitness(i)) && (rand<A1) 
                                       Positions(i,:)=z(i,:);
                                        fitness(i)=Fnew;
                                    end

                elseif abs(A)<1
                    Q(i)=Qmin+(Qmin-Qmax)*rand;
                    v(i,:)=v(i,j)+(Positions(i,:)-Leader_pos(j))*Q(i);
                    z(i,:)= Positions(i,:)+ v(i,:);
                          
                            %%%% problem 
                                if rand>r
                                % The factor 0.001 limits the step sizes of random walks 
                                z(i,:)=Leader_pos(j)+0.001*randn(1,D);
                                end
                                   % Evaluate new solutions
                                   Fnew=feval(fhd,z(i,:)',varargin{:});
                                   % Update if the solution improves, or not too loud
                                if (Fnew<=fitness(i)) && (rand<A1) 
                                   Positions(i,:)=z(i,:);
                                   fitness(i)=Fnew;
                                end
                end
                
            elseif p>=0.5
              
                distance2Leader=abs(Leader_pos(j)-Positions(i,j));
                % Eq. (2.5)
                Positions(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);
                
            end
            
        end
    end
    Tp=Positions(i,:)>VRmax(i,:);Tm=Positions(i,:)<VRmin(i,:);Positions(i,:)=(Positions(i,:).*(~(Tp+Tm)))+VRmax(i,:).*Tp+VRmin(i,:).*Tm;
   
    t=t+1;  
    Convergence_best(t)=Leader_score;
    Convergence_mean(t)=mean(fitness);
end

end




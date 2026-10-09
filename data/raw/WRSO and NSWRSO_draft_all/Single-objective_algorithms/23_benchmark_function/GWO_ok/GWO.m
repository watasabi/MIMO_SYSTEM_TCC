%___________________________________________________________________%
%  Grey Wolf Optimizer (GWO) source codes version 1.0               %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili, S. M. Mirjalili, A. Lewis             %
%               Grey Wolf Optimizer, Advances in Engineering        %
%               Software , in press,                                %
%               DOI: 10.1016/j.advengsoft.2013.12.007               %
%                                                                   %
%___________________________________________________________________%
% Grey Wolf Optimizer
function [Alpha_score,Alpha_pos,Convergence_best,Convergence_mean]=GWO(pop_initial,SearchAgents_no,Max_iter,lb,ub,dim,fobj)
% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems
Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems
Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems
%Initialize the positions of search agents
Positions=pop_initial;
Convergence_best = zeros(1,Max_iter);
Convergence_mean = zeros(1,Max_iter);
fitness = zeros(1,SearchAgents_no);
l=0;% Loop counter
% Main loop
while l<Max_iter
       
        Flag4ub=Positions>ub;
        Flag4lb=Positions<lb;
         Positions=(Positions.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;   
         
   for i=1:SearchAgents_no   
        fitness(i)=fobj(Positions(i,:));
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
    
    a=2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
        for i=1:SearchAgents_no
        for j=1:dim     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7) 
        end
        end
    l=l+1;    
    Convergence_best(l)=Alpha_score;
    Convergence_mean(l)=mean(fitness);
end

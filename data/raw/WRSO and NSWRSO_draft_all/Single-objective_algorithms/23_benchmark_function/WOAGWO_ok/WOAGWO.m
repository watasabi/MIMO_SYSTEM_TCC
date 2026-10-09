%WOAGWO 
% Programmer Hardi M. Mohammed
% A Novel Hybrid GWO with WOA for Global Numerical Optimization
% Hardi M. Mohammed, and Tarik A. Rashid,
%Cite as:                                                         
%   H. Mohammed , T. Rashid. A novel hybrid GWO with WOA for global numerical optimization %and solving pressure vessel design,         
%              Neural Computing and Applications (2020),             
%               DOI: https://doi.org/10.1007/s00521-020-04823-9   
%WOAGWO hybridization is an improvement in Whale Optimization Algorithm
%which includes the hybridization of WOA and GWO algorithms, 
%the results shows that WOAGWO is better than WOA in cec2005, cec2019 and 23 benchmark functions.

%% % WOAGWO modification source codes by % % Hardi M. Mohammed  % % % %
% % we improved the code of WOA which have been written by mirjalili % % then we hybridize GWO algorithm with WOA. % %%
% disclaimer CODE of WOA and GWO are taken from mirjalili website project
% code which is: http://www.alimirjalili.com/Projects.html 
%_________________________________________________________________________%

% The Whale Optimization Algorithm
function [Alpha_score,Alpha_pos,Convergence_best,Convergence_mean]=WOAGWO(pop_initial,SearchAgents_no,Max_iter,lb,ub,dim,fobj)

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems
 
Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems
 
Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=pop_initial;
z=zeros(SearchAgents_no,dim);
fitness = zeros(1,SearchAgents_no);
Convergence_best = zeros(1,Max_iter);
Convergence_mean = zeros(1,Max_iter);
t=0;% Loop counter
while t<Max_iter
    for i=1:SearchAgents_no
        
        % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate objective function for each search agent
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
    
    a=2-t*((2)/Max_iter); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+t*((-1)/Max_iter);
    
    % Update the Position of search agents 
    for i=1:SearchAgents_no
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper
        
        
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
        
       p = rand();        % p in Eq. (2.6)
       
        for j=1:dim
            
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
                    rand_leader_index = floor(SearchAgents_no*rand()+1);
                    X_rand = Positions(rand_leader_index, :);
                    D_X_rand=abs(C*X_rand(j)-Positions(i,j)); 
                    z(i,:)=X_rand(j)-A*D_X_rand;
              
                     % Evaluate new solutions
                                    Fnew=fobj(z(i,:));
%                                      Update if the solution improves, or not too loud
                                    if (Fnew<=fitness(i)) 
                                       Positions(i,:)=z(i,:);
                                        fitness(i)=Fnew;
                                    end
                    
                elseif abs(A)<1
                    D_Leader=abs(C*Alpha_pos(j)-Positions(i,j)); 
                    z(i,:)=Alpha_pos(j)-A*D_Leader;
%                      Evaluate new solutions
                                    Fnew=fobj(z(i,:));
%                                      Update if the solution improves, or not too loud
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

    end
    t=t+1;
    Convergence_best(t)=Alpha_score;
    Convergence_mean(t)=mean(fitness);

end


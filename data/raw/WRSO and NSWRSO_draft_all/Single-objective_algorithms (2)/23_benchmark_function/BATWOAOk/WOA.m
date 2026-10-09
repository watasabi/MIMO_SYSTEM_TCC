%_________________________________________________________________________%
%_________________________________________________________________________%
%WOA-BAT modification

%WOA-BAT modification is an improvement in Whale Optimization Algorithm
%which includes the hybridization of WOA and BAT algorithms, 
%the results shows that WOA-BAT is better than WOA in cec2005, cec2019 and 23 benchmark functions.

%% % WOA-BAT modification source codes by % % Hardi M. Mohammed , Shahla U. Umar, Tarik A. Rashid % % % %
% % we improved the code of WOA which have been written by mirjalili % % then we hybridize BAT algorithm with WOA. % %%
%_________________________________________________________________________%


% The Whale Optimization Algorithm
function [Leader_score,Leader_pos,Convergence_best,Convergence_mean]=WOA(pop_initial,SearchAgents_no,Max_iter,lb,ub,dim,fobj)

% initialize position vector and score for the leader
Leader_pos=zeros(1,dim);
Leader_score=inf; %change this to -inf for maximization problems


%Initialize the positions of search agents
Positions=pop_initial;

Convergence_best = zeros(1,Max_iter);
Convergence_mean = zeros(1,Max_iter);

%bat algorithm addition
Qmin=0;         % Frequency minimum
Qmax=2;         % Frequency maximum
Q=zeros(SearchAgents_no,1);   % Frequency
v=zeros(SearchAgents_no,dim);   % Velocities
r=0.5;
A1=0.5;
t=0;% Loop counter
% summ=0;
% Main loop
fitness = zeros(1,SearchAgents_no);
while t<Max_iter
   Flag4ub=Positions > ub;
   Flag4lb=Positions < lb;
   Positions = (Positions.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; 
    for i=1:SearchAgents_no      
        % Calculate objective function for each search agent
        fitness(i)=fobj(Positions(i,:));
        % Update the leader
        if fitness(i)<Leader_score % Change this to > for maximization problem
            Leader_score=fitness(i); % Update alpha
            Leader_pos=Positions(i,:);
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
            
            if p<0.5   
                
                if abs(A)>=1

                             rand_leader_index = floor(SearchAgents_no*rand()+1);
                             X_rand = Positions(rand_leader_index, :);
                             Q(i)=Qmin+(Qmin-Qmax)*rand;
                             v(i,:)=v(i,j)+(X_rand(j)-Leader_pos(j))*Q(i);
                            z(i,:)= Positions(i,:)+ v(i,:);
                            
                            
                            %%%% problem 
                                if rand>r
                                % The factor 0.001 limits the step sizes of random walks 
                                z (i,:)=Leader_pos(j)+0.001*randn(1,dim);
                                end
                                     % Evaluate new solutions
                                    Fnew=fobj(z(i,:));
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
                                z (i,:)=Leader_pos(j)+0.001*randn(1,dim);
                                end
                                     % Evaluate new solutions
                                    Fnew=fobj(z(i,:));
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
    t=t+1;
    Convergence_best(t)=Leader_score;
    Convergence_mean(t)=mean(fitness);
    
end




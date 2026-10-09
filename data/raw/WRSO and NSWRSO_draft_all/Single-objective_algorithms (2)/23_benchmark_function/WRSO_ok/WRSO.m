%_________________________________________________________________________%
%  Whale Optimization Algorithm (WOA) source codes demo 1.0               %
%                                                                         %
%  Developed in MATLAB R2011b(7.13)                                       %
%                                                                         %
%  Author and programmer: Seyedali Mirjalili                              %
%                                                                         %
%         e-Mail: ali.mirjalili@gmail.com                                 %
%                 seyedali.mirjalili@griffithuni.edu.au                   %
%                                                                         %
%       Homepage: http://www.alimirjalili.com                             %
%                                                                         %
%   Main paper: S. Mirjalili, A. Lewis                                    %
%               The Whale Optimization Algorithm,                         %
%               Advances in Engineering Software , in press,              %
%               DOI: http://dx.doi.org/10.1016/j.advengsoft.2016.01.008   %
%                                                                         %
%_________________________________________________________________________%

% The Whale Optimization Algorithm
function [Leader_score,Leader_pos,Convergence_best,Convergence_mean]=WRSO(pop_initial,SearchAgents_no,Max_iter,lb,ub,dim,fobj)
% initialize position vector and score for the leader
Leader_pos=zeros(1,dim);
Leader_score=inf; %change this to -inf for maximization problems
%Initialize the positions of search agents
Positions=pop_initial;
Convergence_best = zeros(1,Max_iter);
Convergence_mean = zeros(1,Max_iter);
fitness = zeros(1,SearchAgents_no);
t=0;% Loop counter
x = 1;
y = 5;
while t<Max_iter
R = floor((y-x).*rand(SearchAgents_no,1) + x);
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
    r1 = rand(SearchAgents_no,1);
    r2 = rand(SearchAgents_no,1);
    A=2*a*r1-a;
    C=2*r2;
    b=1;               %  parameters in Eq. (2.5)
    l=(a2-1)*rand(SearchAgents_no,1)+1;   %  parameters in Eq. (2.5)
    p = rand(SearchAgents_no,1);       % p in Eq. (2.6)
    rand_leader_index = floor(SearchAgents_no*rand(SearchAgents_no,1)+1);
    X_rand = Positions(rand_leader_index, :);
    D_X_rand=abs((C*ones(1,dim)).*X_rand - Positions); % Eq. (2.7)
    Positions1 = X_rand - (A*ones(1,dim)).*D_X_rand;     
    Ar = R - t*((R)/Max_iter); 
    Cr = 2*rand(SearchAgents_no,dim);
    P_vec=Ar.*Positions + abs(Cr.*(ones(SearchAgents_no,1)*Leader_pos-Positions));                   
    Positions2=ones(SearchAgents_no,1)*Leader_pos-P_vec;
    distance2Leader=abs(ones(SearchAgents_no,1)*Leader_pos-Positions);
    Positions3 = distance2Leader.*exp(b.*l).*cos(l.*2*pi)+ones(SearchAgents_no,1)*Leader_pos;
 
    Positions(p<0.5 & abs(A)>=1,:) = Positions1(p<0.5 & abs(A)>=1,:);            
    Positions(p<0.5 & abs(A)<1,:) = Positions2(p<0.5 & abs(A)<1,:);  
    Positions(p>=0.5,:) = Positions3(p>=0.5,:); 

    t=t+1;
    Convergence_best(t)=Leader_score;
    Convergence_mean(t)=mean(fitness);
end




%_________________________________________________________________________%
%  Rat Swarm Optimizer (RSO)                                              %
%                                                                         %
%  Developed in MATLAB R2019b                                             %
%                                                                         %
%  Designed and Developed: Dr. Gaurav Dhiman                              %
%                                                                         %
%         E-Mail: gdhiman0001@gmail.com                                   %
%                 gaurav.dhiman@ieee.org                                  %
%                                                                         %
%       Homepage: http://www.dhimangaurav.com                             %
%                                                                         %
%  Published paper: G. Dhiman et al.                                      %
%          A novel algorithm for global optimization: Rat Swarm Optimizer %
%               Jounral of Ambient Intelligence and Humanized Computing   %
%               DOI: https://doi.org/10.1007/s12652-020-02580-0           %
%                                                                         %
%_________________________________________________________________________%
function[Leader_score,Leader_pos,Convergence_best,Convergence_mean]=RSO(pop_initial,SearchAgents_no,Max_iterations,Lower_bound,Upper_bound,dim,objective)
Leader_pos=zeros(1,dim);
Leader_score=inf; 
Positions=pop_initial;
Convergence_best = zeros(1,Max_iterations);
Convergence_mean = zeros(1,Max_iterations);
fitness = zeros(1,SearchAgents_no);
l=0;
x = 1;
y = 5;
R = floor((y-x).*rand(1,1) + x);
while l<Max_iterations
   
   Flag4ub=Positions > Upper_bound;
   Flag4lb=Positions < Lower_bound;
   Positions = (Positions.*(~(Flag4ub+Flag4lb)))+Upper_bound.*Flag4ub+Lower_bound.*Flag4lb;
    
    for i=1:SearchAgents_no  
        fitness(i)=objective(Positions(i,:));
        if fitness(i)<Leader_score
            Leader_score=fitness(i); 
            Leader_pos=Positions(i,:);
        end
    end
    A=R-l*((R)/Max_iterations); 
    
    
    for i=1:SearchAgents_no
        for j=1:dim   
            C=2*rand();          
            P_vec=A*Positions(i,j)+abs(C*((Leader_pos(j)-Positions(i,j))));                   
            P_final=Leader_pos(j)-P_vec;
            Positions(i,j)=P_final;
        end
    end
    l=l+1;    
    Convergence_best(l)=Leader_score;
    Convergence_mean(l)=mean(fitness);
end

function [Archive_X,Archive_F] = NSGAII(Ue,Ye,n_obj,search_agents,Max_Gen,dimension)

Ne=dimension;
Np=search_agents;

nVar=Ne;
VarMin= -1;
VarMax= 1;
VarSize=[1 nVar];

lb = VarMin*ones(nVar,1);          % Lower Bound of Variables
ub = VarMax*ones(nVar,1);          % Upper Bound of Variables
% Number of Objective Functions
%% NSGA-II Parameters
MaxIt=Max_Gen;      % Maximum Number of Iterations
nPop=Np;        % Population Size
pCrossover=0.7;                         % Crossover Percentage
nCrossover=2*round(pCrossover*nPop/2);  % Number of Parnets (Offsprings)
pMutation=0.4;                          % Mutation Percentage
nMutation=round(pMutation*nPop);        % Number of Mutants
mu=0.2;                    % Mutation Rate
sigma=0.01*(VarMax-VarMin);  % Mutation Step Size
%% Initialization
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.Rank=[];
empty_individual.DominationSet=[];
empty_individual.DominatedCount=[];
empty_individual.CrowdingDistance=[];
pop=repmat(empty_individual,nPop,1);
for i=1:nPop
    
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost = fun_objetivo(pop(i).Position',Ue,Ye,1);
end
% Non-Dominated Sorting
[pop, F]=NonDominatedSorting(pop);
% Calculate Crowding Distance
pop=CalcCrowdingDistance(pop,F);
% Sort Population
[pop, F]=SortPopulation(pop);
%% NSGA-II Main Loop
for it=1:MaxIt
    
    % Boundary checking
    for   i=1:nPop
    pop(i).Position=min(max(pop(i).Position,lb'),ub');    
    pop(i).Cost=fun_objetivo(pop(i).Position',Ue,Ye,1);
    end

    % Crossover
    popc=repmat(empty_individual,nCrossover/2,2);
    for k=1:nCrossover/2
        
        i1=randi([1 nPop]);
        p1=pop(i1);
        
        i2=randi([1 nPop]);
        p2=pop(i2);
        
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        
        popc(k,1).Cost= fun_objetivo(popc(k,1).Position',Ue,Ye,1);
        popc(k,2).Cost= fun_objetivo(popc(k,2).Position',Ue,Ye,1);
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_individual,nMutation,1);
    for k=1:nMutation
        
        i=randi([1 nPop]);
        p=pop(i);
        
        popm(k).Position=Mutate(p.Position',mu,sigma)';
        
        popm(k).Cost=fun_objetivo(popm(k).Position',Ue,Ye,1);
        
    end
    
    % Merge
    pop=[pop
         popc
         popm]; %#ok
     
     
     
    % Non-Dominated Sorting
    [pop, F]=NonDominatedSorting(pop);
    % Calculate Crowding Distance
    pop=CalcCrowdingDistance(pop,F);
    % Sort Population
    pop=SortPopulation(pop);
    
    % Truncate
    pop=pop(1:nPop);
    
    % Non-Dominated Sorting
    [pop, F]=NonDominatedSorting(pop);
    % Calculate Crowding Distance
    pop=CalcCrowdingDistance(pop,F);
    % Sort Population
    [pop, F]=SortPopulation(pop);
    
    % Store F1
    F1=pop(F{1}); 
end
[num_sol_n_dominadas,comp]=size(F1);
for i = 1:num_sol_n_dominadas
Archive_F(:,i) = F1(i).Cost;
Archive_X(:,i) = F1(i).Position';
end

end


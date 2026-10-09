function [Archive_X,Archive_F] = NSGAII(pop_initial,params,MultiObj)

%% Problem Definition
fun     = MultiObj.fun;
var_min = MultiObj.var_min(:);
var_max = MultiObj.var_max(:);
obj_no=MultiObj.M;
dim=MultiObj.nVar;
VarSize=[1 dim]; 

% Number of Objective Functions
nObj = obj_no;
%% NSGA-II Parameters
MaxIt=params.maxgen;      % Maximum Number of Iterations
nPop=params.Np;        % Population Size
pCrossover=0.7;                         % Crossover Percentage
nCrossover=2*round(pCrossover*nPop/2);  % Number of Parnets (Offsprings)
pMutation=0.4;                          % Mutation Percentage
nMutation=round(pMutation*nPop);        % Number of Mutants
mu=0.02;                    % Mutation Rate
sigma=0.1*(var_max - var_min);  % Mutation Step Size
%% Initialization
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.Rank=[];
empty_individual.DominationSet=[];
empty_individual.DominatedCount=[];
empty_individual.CrowdingDistance=[];
pop=repmat(empty_individual,nPop,1);

POP= pop_initial;
POP_fitness  = fun(POP',MultiObj);

for i=1:nPop    
    pop(i).Position=POP(:,i)';
    pop(i).Cost= POP_fitness(i,:);
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
    pop(i).Position=checkBoundaries(pop(i).Position,var_max,var_min);
    pop(i).Cost=fun(pop(i).Position,MultiObj);
    end

    % Crossover
    popc=repmat(empty_individual,nCrossover/2,2);
    for k=1:nCrossover/2
        
        i1=randi([1 nPop]);
        p1=pop(i1);
        
        i2=randi([1 nPop]);
        p2=pop(i2);
        
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        
        popc(k,1).Cost=fun(popc(k,1).Position,MultiObj);
        popc(k,2).Cost=fun(popc(k,2).Position,MultiObj);
        
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_individual,nMutation,1);
    for k=1:nMutation
        
        i=randi([1 nPop]);
        p=pop(i);
        
        popm(k).Position=Mutate(p.Position,mu,sigma);
        popm(k).Cost=fun(popm(k).Position,MultiObj);
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
Archive_F(i,:) = F1(i).Cost';
Archive_X(i,:) = F1(i).Position;
end

end


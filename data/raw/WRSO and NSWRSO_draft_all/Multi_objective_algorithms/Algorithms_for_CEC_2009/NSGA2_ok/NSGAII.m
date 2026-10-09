function [Archive_X,Archive_F] = NSGAII(pop_initial,dim,obj_no,TestProblem,Max_iteration,SearchAgents_no)

%% Problem Definition
fobj = cec09(TestProblem);
xrange = xboundary(TestProblem, dim);
lb=xrange(:,1)';
ub=xrange(:,2)';
VarSize=[1 dim]; 

% % CostFunction=@(x) MOP4(x);      % Cost Function
% % nVar=3;             % Number of Decision Variables
% % VarSize=[1 nVar];   % Size of Decision Variables Matrix
VarMin= lb;          % Lower Bound of Variables
VarMax= ub;          % Upper Bound of Variables
% Number of Objective Functions
nObj = obj_no;
%% NSGA-II Parameters
MaxIt=Max_iteration;      % Maximum Number of Iterations
nPop=SearchAgents_no;        % Population Size
pCrossover=0.7;                         % Crossover Percentage
nCrossover=2*round(pCrossover*nPop/2);  % Number of Parnets (Offsprings)
pMutation=0.4;                          % Mutation Percentage
nMutation=round(pMutation*nPop);        % Number of Mutants
mu=0.02;                    % Mutation Rate
sigma=0.1*(VarMax-VarMin);  % Mutation Step Size
%% Initialization
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.Rank=[];
empty_individual.DominationSet=[];
empty_individual.DominatedCount=[];
empty_individual.CrowdingDistance=[];
pop=repmat(empty_individual,nPop,1);
for i=1:nPop
    
    pop(i).Position=pop_initial(:,i)';
    %pop(i).Cost=CostFunction(pop(i).Position);
    pop(i).Cost= fobj(pop(i).Position');
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
    pop(i).Position=min(max(pop(i).Position,lb),ub);    
    pop(i).Cost=fobj(pop(i).Position');
    end

    % Crossover
    popc=repmat(empty_individual,nCrossover/2,2);
    for k=1:nCrossover/2
        
        i1=randi([1 nPop]);
        p1=pop(i1);
        
        i2=randi([1 nPop]);
        p2=pop(i2);
        
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        
        popc(k,1).Cost= fobj(popc(k,1).Position');
        popc(k,2).Cost= fobj(popc(k,2).Position');
        
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_individual,nMutation,1);
    for k=1:nMutation
        
        i=randi([1 nPop]);
        p=pop(i);
        
        popm(k).Position=Mutate(p.Position,mu,sigma);
        
        popm(k).Cost=fobj(popm(k).Position');
        
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
    
%     % Show Iteration Information
%     disp(['Iteration ' num2str(it) ': Number of F1 Members = ' num2str(numel(F1))]);
%     
%     % Plot F1 Costs
%     figure(1);
%     PlotCosts(F1);
%     pause(0.01);
    
end
[num_sol_n_dominadas,comp]=size(F1);
for i = 1:num_sol_n_dominadas
Archive_F(:,i) = F1(i).Cost;
Archive_X(i,:) = F1(i).Position;
end

end


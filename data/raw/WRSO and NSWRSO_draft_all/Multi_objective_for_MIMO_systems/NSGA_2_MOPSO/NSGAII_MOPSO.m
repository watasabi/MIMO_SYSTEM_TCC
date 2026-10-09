function [Archive_X,Archive_F] = NSGAII_MOPSO(Ue,Ye,obj_no,SearchAgents_no,Max_iteration,dim)

%% Problem Definition
VarSize=[1 dim]; 
nVar = dim;
VarMin= -1;          % Lower Bound of Variables
VarMax= 1;          % Upper Bound of Variables
lb = VarMin*ones(nVar,1);          % Lower Bound of Variables
ub = VarMax*ones(nVar,1);          % Upper Bound of Variables

% Number of Objective Functions
nObj = obj_no;
%% NSGA-II Parameters
MaxIt=Max_iteration;      % Maximum Number of Iterations
nPop=SearchAgents_no;        % Population Size
pCrossover=0.7;                         % Crossover Percentage
nCrossover=2*round(pCrossover*nPop/2);  % Number of Parnets (Offsprings)
nCrossover=round(nCrossover/2);
pMutation=0.4;                          % Mutation Percentage
nMutation=round(pMutation*nPop);        % Number of Mutants
nMutation=round(nMutation/2);
mug=0.02;                    % Mutation Rate
sigma=0.1*(VarMax-VarMin);  % Mutation Step Size
%% MOPSO Parameters
% MaxIt=2;           % Maximum Number of Iterations
% nPop=length(popul1);            % Population Size
nRep=50;            % Repository Size
w=0.5;              % Inertia Weight
wdamp=0.99;         % Intertia Weight Damping Rate
c1=4;               % Personal Learning Coefficient
c2=1;               % Global Learning Coefficient
nGrid=7;            % Number of Grids per Dimension
alpha=0.1;          % Inflation Rate
beta=2;             % Leader Selection Pressure
gamma=2;            % Deletion Selection Pressure
mu=0.1;             % Mutation Rate

%% Initialization
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.Rank=[];
empty_individual.DominationSet=[];
empty_individual.DominatedCount=[];
empty_individual.CrowdingDistance=[];
pop=repmat(empty_individual,nPop,1);
%% Initialization
empty_particle.Position=[];
empty_particle.Velocity=[];
empty_particle.Cost=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.IsDominated=[];
empty_particle.GridIndex=[];
empty_particle.GridSubIndex=[];
popul=repmat(empty_particle,nPop/2,1);

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
% Truncate and divide into two halves
popul1=pop((nPop/2)+1:nPop);
pop=pop(1:nPop/2);

%% NSGA-II Main Loop
for it=1:MaxIt
    
    % Boundary checking
    for   i=1:nPop/2
    pop(i).Position=min(max(pop(i).Position,lb'),ub');    
    pop(i).Cost=fun_objetivo(pop(i).Position',Ue,Ye,1);
    end
    % Crossover
    popc=repmat(empty_individual,round(nCrossover/2),2);
    for k=1:round(nCrossover/2)
        
        i1=randi([1 nPop/2]);
        p1=pop(i1);
        
        i2=randi([1 nPop/2]);
        p2=pop(i2);
        
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position);

        popc(k,1).Cost= fun_objetivo(popc(k,1).Position',Ue,Ye,1);
        popc(k,2).Cost= fun_objetivo(popc(k,2).Position',Ue,Ye,1);
        
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_individual,nMutation,1);
    for k=1:nMutation
        
        i=randi([1 nPop/2]);
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
    pop=pop(1:nPop/2);
    
    % Non-Dominated Sorting
    [pop, F]=NonDominatedSorting(pop);
    % Calculate Crowding Distance
    pop=CalcCrowdingDistance(pop,F);
    % Sort Population
    [pop, F]=SortPopulation(pop);
    
    % Store F1
    F1=pop(F{1});
    
    
    %% MOPSO
    for i=1:nPop/2
        popul(i).Position=popul1(i).Position; 

        popul(i).Velocity=zeros(VarSize);

        popul(i).Cost=popul1(i).Cost;


        % Update Personal Best
        popul(i).Best.Position=popul(i).Position;
        popul(i).Best.Cost=popul(i).Cost;

    end
    % Determine Domination
    popul=DetermineDomination(popul);
    rep=popul(~[popul.IsDominated]);
    Grid=CreateGrid(rep,nGrid,alpha);
    for i=1:numel(rep)
        rep(i)=FindGridIndex(rep(i),Grid);
    end
%% MOPSO Main Loop
% for it=1:MaxIt
    
    for i=1:nPop/2
        
        leader=SelectLeader(rep,beta);
        
        popul(i).Velocity = w*popul(i).Velocity ...
            +c1*rand(VarSize).*(popul(i).Best.Position-popul(i).Position) ...
            +c2*rand(VarSize).*(leader.Position-popul(i).Position);
        for j=1:nVar
            popul(i).Position(j) = popul(i).Position(j) + popul(i).Velocity(j);
        end
        
        popu1(i).Cost=fun_objetivo(popul(i).Position',Ue,Ye,1);
        % Apply Mutation
        pm=(1-(it-1)/(MaxIt-1))^(1/mu);
        if rand<pm
            NewSol.Position=Mutatetf1(popul(i).Position,pm,lb,ub);
%             NewSol.Position=lchecktf1(NewSol.Position);
            NewSol.Cost=fun_objetivo(NewSol.Position',Ue,Ye,1);
            if Dominates(NewSol,popul(i))
                popul(i).Position=NewSol.Position;
                popul(i).Cost=NewSol.Cost;
            elseif Dominates(popul(i),NewSol)
                % Do Nothing
            else
                if rand<0.5
                    popul(i).Position=NewSol.Position;
                    popul(i).Cost=NewSol.Cost;
                end
            end
        end
        
        if Dominates(popul(i),popul(i).Best)
            popul(i).Best.Position=popul(i).Position;
            popul(i).Best.Cost=popul(i).Cost;
            
        elseif Dominates(popul(i).Best,popul(i))
            % Do Nothing
            
        else
            if rand<0.5
                popul(i).Best.Position=popul(i).Position;
                popul(i).Best.Cost=popul(i).Cost;
            end
        end
        
    end
    
    % Add Non-Dominated Particles to REPOSITORY
    rep=[rep
         popul(~[popul.IsDominated])]; %#ok
    
    % Determine Domination of New Resository Members
    rep=DetermineDomination(rep);
    
    % Keep only Non-Dminated Memebrs in the Repository
    rep=rep(~[rep.IsDominated]);
    
    % Update Grid
%     Grid=CreateGrid(rep,nGrid,alpha);
    % Update Grid Indices
    for i=1:numel(rep)
        rep(i)=FindGridIndex(rep(i),Grid);
    end
%     
%     % Check if Repository is Full
    if numel(rep)>nRep
        
        Extra=numel(rep)-nRep;
        for e=1:Extra
            rep=DeleteOneRepMemebr(rep,gamma);
        end
        
    end

%     % Damping Inertia Weight
    rep1(it)=numel(rep);
    w=w*wdamp;
    pop1=repmat(empty_individual,nPop/2,1);
    for i=1:nPop/2
        if i<=length(rep)
            pop1(i).Position=rep(i).Position;
            pop1(i).Cost=rep(i).Cost;
        else
            pop1(i).Position=popul(i-length(rep)).Position;
            pop1(i).Cost=popul(i-length(rep)).Cost;
        end
    end
    pop2=[pop
          pop1
         ];
    % Non-Dominated Sorting
    [pop2, F]=NonDominatedSorting(pop2);
    % Calculate Crowding Distance
    pop2=CalcCrowdingDistance(pop2,F);
    % Sort Population
    [pop2, F]=SortPopulation(pop2);
    % Truncate and divide into two halves
%     popul1=pop((nPop/2)+1:nPop)
    pop=pop2(1:nPop/2);
% end

end
[num_sol_n_dominadas,comp]=size(F1);
for i = 1:num_sol_n_dominadas
Archive_F(:,i) = F1(i).Cost;
Archive_X(:,i) = F1(i).Position;
end

end


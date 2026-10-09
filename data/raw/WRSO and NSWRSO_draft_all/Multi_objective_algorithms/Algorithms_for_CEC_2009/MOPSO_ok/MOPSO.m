function [Archive_X,Archive_F] = MOPSO(pop_initial,dim,obj_no,TestProblem,Max_iteration,SearchAgents_no)

%% Problem Definition
fobj = cec09(TestProblem);
xrange = xboundary(TestProblem, dim);
lb=xrange(:,1)';
ub=xrange(:,2)';
VarSize=[1 dim]; 

VarMin= lb;          % Lower Bound of Variables
VarMax= ub;          % Upper Bound of Variables
% Number of Objective Functions
nObj = obj_no;
%% MOPSO Parameters
MaxIt=Max_iteration;           % Maximum Number of Iterations
nPop=SearchAgents_no;            % Population Size
nRep=SearchAgents_no;            % Repository Size
w=0.5;              % Inertia Weight
wdamp=0.99;         % Intertia Weight Damping Rate
c1=1;               % Personal Learning Coefficient
c2=2;               % Global Learning Coefficient
nGrid=7;            % Number of Grids per Dimension
alpha=0.1;          % Inflation Rate
beta=2;             % Leader Selection Pressure
gamma=2;            % Deletion Selection Pressure
mu=0.1;             % Mutation Rate
%% Initialization
empty_particle.Position=[];
empty_particle.Velocity=[];
empty_particle.Cost=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.IsDominated=[];
empty_particle.GridIndex=[];
empty_particle.GridSubIndex=[];
pop=repmat(empty_particle,nPop,1);

for i=1:nPop
    
    pop(i).Position=pop_initial(:,i)';
     pop(i).Velocity=zeros(VarSize);
    %pop(i).Cost=CostFunction(pop(i).Position);
    pop(i).Cost= fobj(pop(i).Position');
    % Update Personal Best
    pop(i).Best.Position=pop(i).Position;
    pop(i).Best.Cost=pop(i).Cost;
end

% Determine Domination
pop=DetermineDomination(pop);
rep=pop(~[pop.IsDominated]);
Grid=CreateGrid(rep,nGrid,alpha);
for i=1:numel(rep)
    rep(i)=FindGridIndex(rep(i),Grid);
end
%% MOPSO Main Loop
for it=1:MaxIt
    
    for i=1:nPop
        
        leader=SelectLeader(rep,beta);
        
        pop(i).Velocity = w*pop(i).Velocity ...
            +c1*rand(VarSize).*(pop(i).Best.Position-pop(i).Position) ...
            +c2*rand(VarSize).*(leader.Position-pop(i).Position);
        
        pop(i).Position = pop(i).Position + pop(i).Velocity;
        
        pop(i).Position = max(pop(i).Position, VarMin);
        pop(i).Position = min(pop(i).Position, VarMax);
        
        pop(i).Cost = fobj(pop(i).Position');
        
        % Apply Mutation
        pm=(1-(it-1)/(MaxIt-1))^(1/mu);
        if rand<pm
            NewSol.Position=Mutate(pop(i).Position,pm,VarMin,VarMax);
            NewSol.Cost=fobj(NewSol.Position');
            if Dominates(NewSol,pop(i))
                pop(i).Position=NewSol.Position;
                pop(i).Cost=NewSol.Cost;
            elseif Dominates(pop(i),NewSol)
                % Do Nothing
            else
                if rand<0.5
                    pop(i).Position=NewSol.Position;
                    pop(i).Cost=NewSol.Cost;
                end
            end
        end
        
        if Dominates(pop(i),pop(i).Best)
            pop(i).Best.Position=pop(i).Position;
            pop(i).Best.Cost=pop(i).Cost;
            
        elseif Dominates(pop(i).Best,pop(i))
            % Do Nothing
            
        else
            if rand<0.5
                pop(i).Best.Position=pop(i).Position;
                pop(i).Best.Cost=pop(i).Cost;
            end
        end
        
    end
    
    % Add Non-Dominated Particles to REPOSITORY
    rep=[rep
         pop(~[pop.IsDominated])]; %#ok
    
    % Determine Domination of New Resository Members
    rep=DetermineDomination(rep);
    
    % Keep only Non-Dminated Memebrs in the Repository
    rep=rep(~[rep.IsDominated]);
    
    % Update Grid
    Grid=CreateGrid(rep,nGrid,alpha);
    % Update Grid Indices
    for i=1:numel(rep)
        rep(i)=FindGridIndex(rep(i),Grid);
    end
    
    % Check if Repository is Full
    if numel(rep)>nRep
        
        Extra=numel(rep)-nRep;
        for e=1:Extra
            rep=DeleteOneRepMemebr(rep,gamma);
        end
        
    end
    
% % %    Plot Costs
% %     figure(1);
% %     PlotCosts(pop,rep);
% %     pause(0.01);
% %     
% % %    Show Iteration Information
% %     disp(['Iteration ' num2str(it) ': Number of Rep Members = ' num2str(numel(rep))]);
% %     
    % Damping Inertia Weight
    w=w*wdamp;
    
end
[num_sol_n_dominadas,comp]=size(rep);
for i = 1:num_sol_n_dominadas
Archive_F(:,i) = rep(i).Cost;
Archive_X(i,:) = rep(i).Position;
end

end


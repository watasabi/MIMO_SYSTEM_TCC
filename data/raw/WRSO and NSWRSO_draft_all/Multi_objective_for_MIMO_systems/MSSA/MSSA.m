function [Archive_X,Archive_F] = MSSA(Ue,Ye,obj_no,SearchAgents_no,Max_iteration,dim)

Ne=dim;
Np=SearchAgents_no;

nVar=Ne;
VarMin = -1;
VarMax = 1;
VarSize=[Np nVar];
lb = VarMin*ones(dim,1);          % Lower Bound of Variables
ub = VarMax*ones(dim,1); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter=Max_iteration;
N=SearchAgents_no;
ArchiveMaxSize=SearchAgents_no;
Archive_X=zeros(N,dim);
Archive_F=ones(N,obj_no)*inf;
Archive_member_no=0;
r=(ub-lb)/2;
V_max=(ub(1)-lb(1))/10;
Food_fitness=inf*ones(1,obj_no);
Food_position=zeros(dim,1);
Salps_X=unifrnd(VarMin,VarMax,VarSize)';
fitness=zeros(N,2);
position_history=zeros(N,max_iter,dim);

for iter=1:max_iter
        
 c1 = 2*exp(-(4*iter/max_iter)^2); % Eq. (3.2) in the paper
    
    for i=1:N %Calculate all the objective values first
        Salps_fitness(i,:)=fun_objetivo(Salps_X(:,i),Ue,Ye,1);
        if dominates(Salps_fitness(i,:),Food_fitness)
            Food_fitness=Salps_fitness(i,:);
            Food_position=Salps_X(:,i);
        end
    end
    
    [Archive_X, Archive_F, Archive_member_no]=UpdateArchive(Archive_X, Archive_F, Salps_X, Salps_fitness, Archive_member_no);
    
    if Archive_member_no>ArchiveMaxSize
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
        [Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no]=HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize);
    else
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    end
    
    Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    % Archive_mem_ranks
    % Chose the archive member in the least population area as food`
    % to improve coverage
    index=RouletteWheelSelection(1./Archive_mem_ranks);
    if index==-1
        index=1;
    end
    Food_fitness=Archive_F(index,:);
    Food_position=Archive_X(index,:)';
    
    for i=1:N
        
        index=0;
        neighbours_no=0;
        
        if i<=N/2
            for j=1:1:dim
                c2=rand();
                c3=rand();
                %%%%%%%%%%%%% % Eq. (3.1) in the paper %%%%%%%%%%%%%%
                if c3<0.5
                    Salps_X(j,i)=Food_position(j)+c1*((ub(j)-lb(j))*c2+lb(j));
                else
                    Salps_X(j,i)=Food_position(j)-c1*((ub(j)-lb(j))*c2+lb(j));
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        elseif i>N/2 && i<N+1
            
            point1=Salps_X(:,i-1);
            point2=Salps_X(:,i);
            
            Salps_X(:,i)=(point2+point1)/(2); % Eq. (3.4) in the paper
        end
        
        Flag4ub=Salps_X(:,i)>ub;
        Flag4lb=Salps_X(:,i)<lb;
        Salps_X(:,i)=(Salps_X(:,i).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
    end
    
end
Archive_F=Archive_F';
Archive_X=Archive_X';
end


function [Archive_X,Archive_F] = MSSA(pop_initial,params,MultiObj)

fun     = MultiObj.fun;
var_min = MultiObj.var_min(:);
lb = var_min;
var_max = MultiObj.var_max(:);
ub =var_max;
obj_no=MultiObj.M;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dim=MultiObj.nVar;
max_iter=params.maxgen;
N=params.Np;
ArchiveMaxSize=params.Nr;
Archive_X=zeros(N,dim);
Archive_F=ones(N,obj_no)*inf;
Archive_member_no=0;
Food_fitness=inf*ones(1,obj_no);
Food_position=zeros(dim,1);
Salps_X = pop_initial;



for iter=1:max_iter
        
    c1 = 2*exp(-(4*iter/max_iter)^2); % Eq. (3.2) in the paper
    Salps_fitness  = fun(Salps_X',MultiObj);
    for i=1:N %Calculate all the objective values first
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
        

        POS = Salps_X';
        [POS] = checkBoundaries(POS,var_max,var_min);
        Salps_X = POS';
    end
    
 %   display(['At the iteration ', num2str(iter), ' there are ', num2str(Archive_member_no), ' non-dominated solutions in the archive']);
    
end


end


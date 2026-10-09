clear all 
clc
load Population_gen.mat
global fname
 N_function=15;% number of test function
 runtimes=50;  % odd number
 hyp = zeros(N_function,runtimes);
 IGDf = zeros(N_function,runtimes);
 time = zeros(N_function,runtimes);
 % Parameters
params.Np = 100;        % Population size
params.Nr = 100;        % Repository size
params.maxgen = 1000;    % Maximum number of generations

for i_func= 1:N_function
        switch i_func
            case 1
                fname='MaF1'; % (Modified inverted DTLZ1)
                ManyObj.M = 5;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun1;
                load PFMaF1.mat
                PF = PFMaF1;
            case 2
                fname='MaF2'; % DTLZ2BZ 
                ManyObj.M = 5;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun2;
                load PFMaF2.mat
                PF = PFMaF2;
            case 3
                fname='MaF3'; % (Convex DTLZ3)
                ManyObj.M = 5;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun3;
                load PFMaF3.mat
                PF = PFMaF3;
            case 4
                fname='MaF4'; % (Inverted badly-scaled DTLZ3)
                ManyObj.M = 5;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun4; 
                load PFMaF4.mat
                PF = PFMaF4;
            case 5
                fname='MaF5'; % (Concave badly-scaled DTLZ4)
                ManyObj.M = 5;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun5;
                load PFMaF5.mat
                PF = PFMaF5;
             case 6
                fname='MaF6'; % DTLZ5(I,M)
                ManyObj.M = 5;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun6;
                load PFMaF6.mat
                PF = PFMaF6;
            case 7
                fname='MaF7';  % DTLZ7
                ManyObj.M = 5;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun7;
                load PFMaF7.mat
                PF = PFMaF7;
            case 8
                fname='MaF8';  %(Multi-Point Distance Minimization Problem
                ManyObj.M = 5; % número de funções objetivas
                ManyObj.nVar = 2;
                ManyObj.var_min = [-10000,-10000];
                ManyObj.var_max = [10000,10000];
                ManyObj.fun = @fun8;
                load PFMaF8.mat
                PF = PFMaF8;
            case 9
                fname='MaF9';  %(Multi-Line Distance Minimization Problem
                ManyObj.M = 5; % número de funções objetivas
                ManyObj.nVar = 2;
                ManyObj.var_min = [-10000,-10000];
                ManyObj.var_max = [10000,10000];
                ManyObj.fun = @fun9;
                load PFMaF9.mat
                PF = PFMaF9;
             case 10
                fname='MaF10';  %(WFG1)
                ManyObj.M = 5; % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = 2:2:2*ManyObj.nVar;
                ManyObj.fun = @fun10;
                load PFMaF10.mat
                PF = PFMaF10;
             case 11
                fname='MaF11';   %(WFG2)
                ManyObj.M = 5; % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.nVar = ceil((ManyObj.nVar-ManyObj.M+1)/2)*2 + ManyObj.M - 1;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = 2:2:2*ManyObj.nVar;
                ManyObj.fun = @fun11;
                load PFMaF11.mat
                PF = PFMaF11;
            case 12 
                fname='MaF12';   %(WFG9)
                ManyObj.M = 5; % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = 2:2:2*ManyObj.nVar;
                ManyObj.fun = @fun12;
                load PFMaF12.mat
                PF = PFMaF12;
            case 13
                fname='MaF13';   %(PF7)
                ManyObj.M = 5; % número de funções objetivas
                ManyObj.nVar = 5;
                ManyObj.var_min = [zeros(1,2),zeros(1,ManyObj.nVar-2)-2];
                ManyObj.var_max = [ones(1,2),zeros(1,ManyObj.nVar-2)+2];
                ManyObj.fun = @fun13;
                load PFMaF13.mat
                PF = PFMaF13;
            case 14 
                fname='MaF14';   %(LSMOP3)
                ManyObj.M = 5; % número de funções objetivas
                ManyObj.nVar = 20*ManyObj.M;
                nk = 2;
                c = 3.8*0.1*(1-0.1);
                for i = 1: ManyObj.M-1
                    c = [c,3.8.*c(end).*(1-c(end))];
                end
                sublen = ceil(round(c./sum(c).*ManyObj.nVar)/nk);
                len    = [0,cumsum(sublen*nk)];
                ManyObj.nVar = ManyObj.M - 1 + len(end);
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = [ones(1,ManyObj.M-1),10.*ones(1,len(end))];
                ManyObj.fun = @fun14;
                load PFMaF14.mat
                PF = PFMaF14;
            case 15 
                fname='MaF15';   %(Inverted LSMOP8)
                ManyObj.M = 5; % número de funções objetivas
                ManyObj.nVar = 20*ManyObj.M;
                nk = 2;
                c = 3.8*0.1*(1-0.1);
                for i = 1: ManyObj.M-1
                    c = [c,3.8.*c(end).*(1-c(end))];
                end
                sublen = ceil(round(c./sum(c).*ManyObj.nVar)/nk);
                len    = [0,cumsum(sublen*nk)];
                ManyObj.nVar = ManyObj.M - 1 + len(end);
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = [ones(1,ManyObj.M-1),10.*ones(1,len(end))];
                ManyObj.fun = @fun15;
                load PFMaF15.mat
                PF = PFMaF15;
        end
        display(fname);
       %% Load reference PS and PF data
       %% Initialize the population size and the maximum evaluations
          popsize=params.Np;
          n_obj=ManyObj.M;
          repoint = ones(1,n_obj);
          PARETO_FRONT=zeros(n_obj,popsize,runtimes);
           for j=1:runtimes
           pop_initial = POP(i_func,j).Position;
           tic
           [Archive_X,Archive_F] = MSSA(pop_initial,params,ManyObj);
           time(i_func,j)=toc;
           % Indicators
            hyp(i_func,j)=Hypervolume_calculation(Archive_F,repoint);
            IGDf(i_func,j)=IGD_calculation(Archive_F,PF);
            
            [num_sol_n_dominadas,comp]=size(Archive_F);

            for k = 1:num_sol_n_dominadas
            PARETO_FRONT(:,k,j) = Archive_F(k,:)'; 
            end
            
            save (fname,'PARETO_FRONT');
           end
 
 end 
metaheuristic = 'MSSA_5_obj';
save(metaheuristic,'hyp','IGDf','time') 
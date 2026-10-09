clc
clear
dim = 10;
Np = 100;
N = 15; %number of functions
r = 50; %number of runs.
empty_individual.Position=[];
pop=repmat(empty_individual,N,r);
for ii=1:N
       switch ii
            case 1
                fname='MaF1'; % (Modified inverted DTLZ1)
                ManyObj.M = 3;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun1;
            case 2
                fname='MaF2'; % DTLZ2BZ 
                ManyObj.M = 3;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun2;
            case 3
                fname='MaF3'; % (Convex DTLZ3)
                ManyObj.M = 3;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun3;
            case 4
                fname='MaF4'; % (Inverted badly-scaled DTLZ3)
                ManyObj.M = 3;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun4; 
                load PFMaF4.mat
                PF = PFMaF4;
            case 5
                fname='MaF5'; % (Concave badly-scaled DTLZ4)
                ManyObj.M = 3;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun5;
             case 6
                fname='MaF6'; % DTLZ5(I,M)
                ManyObj.M = 3;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun6;
            case 7
                fname='MaF7';  % DTLZ7
                ManyObj.M = 3;    % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = ones(1,ManyObj.nVar);
                ManyObj.fun = @fun7;
            case 8
                fname='MaF8';  %(Multi-Point Distance Minimization Problem
                ManyObj.M = 3; % número de funções objetivas
                ManyObj.nVar = 2;
                ManyObj.var_min = [-10000,-10000];
                ManyObj.var_max = [10000,10000];
                ManyObj.fun = @fun8;
            case 9
                fname='MaF9';  %(Multi-Line Distance Minimization Problem
                ManyObj.M = 3; % número de funções objetivas
                ManyObj.nVar = 2;
                ManyObj.var_min = [-10000,-10000];
                ManyObj.var_max = [10000,10000];
                ManyObj.fun = @fun9;
             case 10
                fname='MaF10';  %(WFG1)
                ManyObj.M = 3; % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = 2:2:2*ManyObj.nVar;
                ManyObj.fun = @fun10;
                load PFMaF10.mat
                PF = PFMaF10;
             case 11
                fname='MaF11';   %(WFG2)
                ManyObj.M = 3; % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.nVar = ceil((ManyObj.nVar-ManyObj.M+1)/2)*2 + ManyObj.M - 1;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = 2:2:2*ManyObj.nVar;
                ManyObj.fun = @fun11;
            case 12 
                fname='MaF12';   %(WFG9)
                ManyObj.M = 3; % número de funções objetivas
                ManyObj.nVar = ManyObj.M + 9;
                ManyObj.var_min = zeros(1,ManyObj.nVar);
                ManyObj.var_max = 2:2:2*ManyObj.nVar;
                ManyObj.fun = @fun12;
            case 13
                fname='MaF13';   %(PF7)
                ManyObj.M = 3; % número de funções objetivas
                ManyObj.nVar = 5;
                ManyObj.var_min = [zeros(1,2),zeros(1,ManyObj.nVar-2)-2];
                ManyObj.var_max = [ones(1,2),zeros(1,ManyObj.nVar-2)+2];
                ManyObj.fun = @fun13;
            case 14 
                fname='MaF14';   %(LSMOP3)
                ManyObj.M = 3; % número de funções objetivas
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
            case 15 
                fname='MaF15';   %(Inverted LSMOP8)
                ManyObj.M = 3; % número de funções objetivas
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
        end
    var_min = ManyObj.var_min';
    var_max = ManyObj.var_max';
    obj_no=ManyObj.M;
    dim =ManyObj.nVar;
        
for j=1:r
    pop(ii,j).Position=(repmat((var_max-var_min)',Np,1).*rand(Np,dim) + repmat(var_min',Np,1))';
end
end
POP=pop;
save('Population_gen','POP')

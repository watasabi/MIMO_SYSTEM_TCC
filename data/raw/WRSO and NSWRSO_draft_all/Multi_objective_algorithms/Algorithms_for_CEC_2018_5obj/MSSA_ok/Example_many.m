% ----------------------------------------------------------------------- %
% Example of use of the funcion Many-Objective %                         %
% ----------------------------------------------------------------------- %
%   Author:  Gideon Villar Leandro                                        %
%   Date:    19/04/2021                                                   %                           %
% ----------------------------------------------------------------------- %
%   References:                                                           %
%    Ran Cheng, Miqing Li, Ye Tian, Xingyi Zhang, Shengxiang Yang,        %
%    Yaochu Jin and Xin Yao " Benchmark Functions for the CEC'2018        %
%    Competition on Many-Objective Optimization", Technical Report,       %
%    University of Birmingham, United Kingdom, 2018.                      %
% ----------------------------------------------------------------------- %
%clear all; clc;

% Parameters
params.Np = 200;        % Population size
params.Nr = 200;        % Repository size
params.maxgen = 1000;    % Maximum number of generations
% params.W = 0.4;         % Inertia weight
% params.C1 = 2;          % Individual confidence factor
% params.C2 = 2;          % Swarm confidence factor
% params.ngrid = 20;      % Number of grids in each dimension
% params.maxvel = 5;      % Maxmium vel in percentage
% params.u_mut = 0.5;     % Uniform mutation percentage

% Multi-objective function
%ManyObjFnc = 'MaF1';
ManyObjFnc = 'MaF2';
%ManyObjFnc = 'MaF3';
%ManyObjFnc = 'MaF4';
%ManyObjFnc = 'MaF5';
%ManyObjFnc = 'MaF6';
%ManyObjFnc = 'MaF7';
%ManyObjFnc = 'MaF8';
%ManyObjFnc = 'MaF9';
%ManyObjFnc = 'MaF10';
%ManyObjFnc = 'MaF11';
%ManyObjFnc = 'MaF12';
%ManyObjFnc = 'MaF13';
%ManyObjFnc = 'MaF14';
%ManyObjFnc = 'MaF15';
switch ManyObjFnc
    case 'MaF1'   % (Modified inverted DTLZ1)
        ManyObj.M = 3;    % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = ones(1,ManyObj.nVar);
        ManyObj.fun = @fun1;
    case 'MaF2'   % DTLZ2BZ 
        ManyObj.M = 3;    % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = ones(1,ManyObj.nVar);
        ManyObj.fun = @fun2;
    case 'MaF3'   % (Convex DTLZ3)
        ManyObj.M = 3;    % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = ones(1,ManyObj.nVar);
        ManyObj.fun = @fun3;
    case 'MaF4'   % (Inverted badly-scaled DTLZ3)
        ManyObj.M = 3;    % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = ones(1,ManyObj.nVar);
        ManyObj.fun = @fun4;
    case 'MaF5'   % (Concave badly-scaled DTLZ4)
        ManyObj.M = 3;    % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = ones(1,ManyObj.nVar);
        ManyObj.fun = @fun5;
    case 'MaF6'   % DTLZ5(I,M)
        ManyObj.M = 3;    % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = ones(1,ManyObj.nVar);
        ManyObj.fun = @fun6;
    case 'MaF7'   % DTLZ7
        ManyObj.M = 3;    % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = ones(1,ManyObj.nVar);
        ManyObj.fun = @fun7;
    case 'MaF8'   %(Multi-Point Distance Minimization Problem
        ManyObj.M = 3; % número de funções objetivas
        ManyObj.nVar = 2;
        ManyObj.var_min = [-10000,-10000];
        ManyObj.var_max = [10000,10000];
        ManyObj.fun = @fun8;
    case 'MaF9'   %(Multi-Line Distance Minimization Problem
        ManyObj.M = 3; % número de funções objetivas
        ManyObj.nVar = 2;
        ManyObj.var_min = [-10000,-10000];
        ManyObj.var_max = [10000,10000];
        ManyObj.fun = @fun9;
    case 'MaF10'   %(WFG1)
        ManyObj.M = 3; % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = 2:2:2*ManyObj.nVar;
        ManyObj.fun = @fun10;
    case 'MaF11'   %(WFG2)
        ManyObj.M = 3; % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.nVar = ceil((ManyObj.nVar-ManyObj.M+1)/2)*2 + ManyObj.M - 1;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = 2:2:2*ManyObj.nVar;
        ManyObj.fun = @fun11;
    case 'MaF12'   %(WFG9)
        ManyObj.M = 3; % número de funções objetivas
        ManyObj.nVar = ManyObj.M + 9;
        ManyObj.var_min = zeros(1,ManyObj.nVar);
        ManyObj.var_max = 2:2:2*ManyObj.nVar;
        ManyObj.fun = @fun12;
    case 'MaF13'   %(PF7)
        ManyObj.M = 3; % número de funções objetivas
        ManyObj.nVar = 5;
        ManyObj.var_min = [zeros(1,2),zeros(1,ManyObj.nVar-2)-2];
        ManyObj.var_max = [ones(1,2),zeros(1,ManyObj.nVar-2)+2];
        ManyObj.fun = @fun13;
    case 'MaF14'   %(LSMOP3)
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
    case 'MaF15'   %(Inverted LSMOP8)
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
% MOPSO
[Archive_X,Archive_F] = MWRSO(params,ManyObj);
plot3(Archive_F(:,1),Archive_F(:,2),Archive_F(:,3),'ro','MarkerSize',8,'markerfacecolor','k')
% Display info
display('Repository fitness values are stored in REP.pos_fit');
display('Repository particles positions are store in REP.pos');

 clear all
 mex cec19_func.cpp -DWINDOWS

load('Population_gen.mat');
N = 10; %number of functions
BEST_SCORE_AVERANGE = zeros(N,1);
BEST_SCORE_STD = zeros(N,1);
TIME_AVERANGE = zeros(N,1);
TIME_STD = zeros(N,1);
pop_size=30;
iter_max=1000;
runs=100;
CONVERGENCE_BEST = zeros(runs,iter_max);
CONVERGENCE_MEAN = zeros(runs,iter_max);
Function_name={'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'}; % Name of the test function that can be from F1 to F10 

fhd=str2func('cec19_func');
for i=1:N
    BEST_SCORE=zeros(1,runs);
    time=zeros(1,runs);
    func_num=i;
    if i == 1
    Xmin=-8192;
    Xmax=8192;
    D=9;
    end
    if i == 2
    Xmin=-16384;
    Xmax=16384;
    D=16;
    end
    if i == 3
    Xmin=-4;
    Xmax= 4;
    D=18;
    end
    if i >= 4
    Xmin=-100;
    Xmax= 100;
    D=10;
    end
    
    for j=1:runs
    pop_initial = POP(i,j).Position; 
        tic
        [gbest,gbestval,CONVERGENCE_BEST(j,:),CONVERGENCE_MEAN(j,:)]= SSA_func(pop_initial,fhd,D,pop_size,iter_max,Xmin,Xmax,func_num);
        time(1,j)=toc; 
        BEST_SCORE(1,j) =gbestval;
    end
    fx = Function_name{1,i}
    save (fx, 'BEST_SCORE','time','CONVERGENCE_BEST','CONVERGENCE_MEAN');
    BEST_SCORE_AVERANGE(i,1) = mean(BEST_SCORE);
    BEST_SCORE_STD(i,1) = std(BEST_SCORE);
    TIME_AVERANGE(i,1) = mean(time);
    TIME_STD(i,1) = std(time);
end
txt = 'Result Table';
save (txt, 'BEST_SCORE_AVERANGE','BEST_SCORE_STD','TIME_AVERANGE','TIME_STD');


clear all 
clc
load('Population_gen.mat');
SearchAgents_no=30; % Number of search agents
Max_iteration=500; % Maximum numbef of iterations
tam=100; %number of runs
N = 23; %number of functions

BEST_SCORE_AVERANGE = zeros(N,1);
BEST_SCORE_STD = zeros(N,1);
TIME_AVERANGE = zeros(N,1);
TIME_STD = zeros(N,1);
CONVERGENCE_BEST = zeros(tam,Max_iteration);
CONVERGENCE_MEAN = zeros(tam,Max_iteration);
Function_name={'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23'}; % Name of the test function that can be from F1 to F23 

for i=1:N
% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Get_Functions_details(Function_name{1,i});
BEST_SCORE=zeros(1,tam);
time=zeros(1,tam);
for k=1:tam
pop_initial = POP(i,k).Position;   
tic    
[Best_score,Best_pos,CONVERGENCE_BEST(k,:),CONVERGENCE_MEAN(k,:)]=WRSO(pop_initial,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);    
time(1,k)=toc;
BEST_SCORE(1,k) = Best_score;
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
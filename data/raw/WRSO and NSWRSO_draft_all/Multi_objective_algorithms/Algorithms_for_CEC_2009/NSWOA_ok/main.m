clear all
clc
load('Population_gen.mat');

load UF1.dat
load UF2.dat
load UF3.dat
load UF4.dat
load UF5.dat
load UF6.dat
load UF7.dat
load UF8.dat
load UF9.dat
load UF10.dat
UF = {UF1,UF2,UF3,UF4,UF5,UF6,UF7,UF8,UF9,UF10};
obj_no= {2,2,2,2,2,2,2,3,3,3};
dim=10;
SearchAgents_no=100; % Number of search agents
Max_iteration=1000; % Maximum numbef of iterations 1000
tam=50; %number of runs
N = 10; %number of functions
IGD_AVERANGE = zeros(N,1);
IGD_STD = zeros(N,1);
HYPER_V_AVERANGE = zeros(N,1);
HYPER_V_STD = zeros(N,1);
TIME_AVERANGE = zeros(N,1);
TIME_STD = zeros(N,1);
Function_name={'UF1','UF2','UF3','UF4','UF5','UF6','UF7','UF8','UF9','UF10'}; % Name of the test function that can be from F1 to F23 
Archive_size=100; 
for i=1:N
IGD_SCORE=zeros(1,tam);
HYPER_V_SCORE=zeros(1,tam);
PARETO_FRONT=zeros(obj_no{i},Archive_size,tam);
TestProblem = Function_name{1,i}    
time=zeros(1,tam);
for k=1:tam
pop_initial = POP(i,k).Position;
tic    
[Archive_X,Archive_F] = NSWOA(pop_initial,dim,obj_no{i},TestProblem,Max_iteration,SearchAgents_no);
a= Archive_F;
time(1,k)=toc;
IGD_SCORE(1,k)=IGD_calculation(Archive_F',UF{i});
repoint = ones(1,obj_no{i});
HYPER_V_SCORE(1,k)=Hypervolume_calculation(Archive_F',repoint);

[comp,num_sol_n_dominadas]=size(Archive_F);

for j = 1:num_sol_n_dominadas
PARETO_FRONT(:,j,k) = Archive_F(:,j); 
end
end

save (TestProblem,'IGD_SCORE','HYPER_V_SCORE','PARETO_FRONT','time');
IGD_AVERANGE(i,1) = mean(IGD_SCORE)
IGD_STD(i,1) = std(IGD_SCORE);
HYPER_V_AVERANGE(i,1) = mean(HYPER_V_SCORE);
HYPER_V_STD(i,1) = std(HYPER_V_SCORE);
TIME_AVERANGE(i,1) = mean(time);
TIME_STD(i,1) = std(time);
end
txt = 'Result Table';
save (txt, 'IGD_AVERANGE','IGD_STD','HYPER_V_AVERANGE','HYPER_V_STD','TIME_AVERANGE','TIME_STD');
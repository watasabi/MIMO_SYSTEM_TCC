clc
clear

SearchAgents_no = 30;
N = 23; %number of functions
r = 100; %number of runs.
Function_name={'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23'}; % Name of the test function that can be from F1 to F23 

empty_individual.Position=[];
pop=repmat(empty_individual,N,r);
for i=1:N
[lb,ub,dim,fobj]=Get_Functions_details(Function_name{1,i});

for j=1:r
    pop(i,j).Position=initialization(SearchAgents_no,dim,ub,lb);
end
end
POP=pop;
save('Population_gen','POP')

clc
clear

ps = 30;
N = 10; %number of functions
r = 100; %number of runs.
Function_name={'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'}; % Name of the test function that can be from F1 to F23 

empty_individual.Position=[];
pop=repmat(empty_individual,N,r);
for i=1:N
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
for j=1:r
    if length(Xmin)==1
    a=repmat(Xmin,1,D);
    b=repmat(Xmax,1,D);
    end
    Xmin=repmat(a,ps,1);
    Xmax=repmat(b,ps,1);
    pop(i,j).Position=a+(b-a).*rand(ps,D);
end
end
POP=pop;
save('Population_gen','POP')

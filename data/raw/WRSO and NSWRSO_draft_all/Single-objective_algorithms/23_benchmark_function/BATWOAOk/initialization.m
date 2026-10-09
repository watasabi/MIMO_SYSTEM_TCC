%_________________________________________________________________________%
%_________________________________________________________________________%
%WOA-BAT modification

%WOA-BAT modification is an improvement in Whale Optimization Algorithm
%which includes the hybridization of WOA and BAT algorithms, 
%the results shows that WOA-BAT is better than WOA in cec2005, cec2019 and 23 benchmark functions.

%% % WOA-BAT modification source codes by % % Hardi M. Mohammed , Shahla U. Umar, Tarik A. Rashid % % % %
% % we improved the code of WOA which have been written by mirjalili % % then we hybridize BAT algorithm with WOA. % %%
%_________________________________________________________________________%


% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
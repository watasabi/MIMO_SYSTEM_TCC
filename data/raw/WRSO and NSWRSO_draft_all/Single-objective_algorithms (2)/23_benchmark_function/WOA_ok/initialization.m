%  WOAGWO source codes demo 1.0                                                                               
%  Developed in MATLAB R2017a                                       
%  Author and programmer: Hardi M. Mohammed                              
%  e-Mail: hardi.mohammed@charmouniversity.org                                                                                                                                         
%       Homepage: https://sites.google.com/a/charmouniversity.org/hardi-mohammed/           
%                       http://www.nci-rc.com/about.php                                                           
%   Main paper: cite as: H. Mohammed , T. Rashid                                                                                                     
%              A novel hybrid GWO with WOA for global numerical optimization and solving pressure vessel design,         
%              Neural Computing and Applications (2020),             
%               DOI: https://doi.org/10.1007/s00521-020-04823-9   

%WOAGWO hybridization is an improvement in Whale Optimization Algorithm
%which includes the hybridization of WOA and GWO algorithms, 
%the results shows that WOAGWO is better than WOA in cec2005, cec2019 and 23 benchmark functions.

%% % WOAGWO modification source codes by % % Hardi M. Mohammed  % % % %
% % we improved the code of WOA which have been written by mirjalili % % then we hybridize GWO algorithm with WOA. % %%
% disclaimer CODE of WOA and GWO are taken from mirjalili website project
% code which is: http://www.alimirjalili.com/Projects.html 
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
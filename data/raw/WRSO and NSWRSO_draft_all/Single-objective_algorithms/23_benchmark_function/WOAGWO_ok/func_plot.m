%  WOAGWO source codes demo 1.0                                                                               
%  Developed in MATLAB R2017a                                       
%  Author and programmer: Hardi M. Mohammed                              
%  e-Mail: hardi.mohammed@charmouniversity.org                                                                                                                                         
%       Homepage: https://sites.google.com/a/charmouniversity.org/hardi-mohammed/           
%                       http://www.nci-rc.com/about.php                                                           
%   Main paper: cite as : H. Mohammed , T. Rashid                                                                                                     
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
% This function draw the benchmark functions

function func_plot(func_name)

[lb,ub,dim,fobj]=Get_Functions_details(func_name);

switch func_name 
    case 'F1' 
        x=-100:2:100; y=x; %[-100,100]
        
    case 'F2' 
        x=-100:2:100; y=x; %[-10,10]
        
    case 'F3' 
        x=-100:2:100; y=x; %[-100,100]
        
    case 'F4' 
        x=-100:2:100; y=x; %[-100,100]
    case 'F5' 
        x=-200:2:200; y=x; %[-5,5]
    case 'F6' 
        x=-100:2:100; y=x; %[-100,100]
    case 'F7' 
        x=-1:0.03:1;  y=x  %[-1,1]
    case 'F8' 
        x=-500:10:500;y=x; %[-500,500]
    case 'F9' 
        x=-5:0.1:5;   y=x; %[-5,5]    
    case 'F10' 
        x=-20:0.5:20; y=x;%[-500,500]
    case 'F11' 
        x=-500:10:500; y=x;%[-0.5,0.5]
    case 'F12' 
        x=-10:0.1:10; y=x;%[-pi,pi]
    case 'F13' 
        x=-5:0.08:5; y=x;%[-3,1]
    case 'F14' 
        x=-100:2:100; y=x;%[-100,100]
    case 'F15' 
        x=-5:0.1:5; y=x;%[-5,5]
    case 'F16' 
        x=-1:0.01:1; y=x;%[-5,5]
    case 'F17' 
        x=-5:0.1:5; y=x;%[-5,5]
    case 'F18' 
        x=-5:0.06:5; y=x;%[-5,5]
    case 'F19' 
        x=-5:0.1:5; y=x;%[-5,5]
    case 'F20' 
        x=-5:0.1:5; y=x;%[-5,5]        
    case 'F21' 
        x=-5:0.1:5; y=x;%[-5,5]
    case 'F22' 
        x=-5:0.1:5; y=x;%[-5,5]     
    case 'F23' 
        x=-5:0.1:5; y=x;%[-5,5] 
        case 'cec10' 
        x=-100:2:100; y=x;%[-100,100]
end    

    

L=length(x);
f=[];

for i=1:L
    for j=1:L
        if strcmp(func_name,'F15')==0 && strcmp(func_name,'F19')==0 && strcmp(func_name,'F20')==0 && strcmp(func_name,'F21')==0 && strcmp(func_name,'F22')==0 && strcmp(func_name,'F23')==0
            f(i,j)=fobj([x(i),y(j)]);
        end
        if strcmp(func_name,'cec10')==1
            f(i,j)=fobj([x(i),y(j),0,0]);
        end
        if strcmp(func_name,'F15')==1
            f(i,j)=fobj([x(i),y(j),0,0]);
        end
        if strcmp(func_name,'F19')==1
            f(i,j)=fobj([x(i),y(j),0]);
        end
        if strcmp(func_name,'F20')==1
            f(i,j)=fobj([x(i),y(j),0,0,0,0]);
        end       
        if strcmp(func_name,'F21')==1 || strcmp(func_name,'F22')==1 ||strcmp(func_name,'F23')==1
            f(i,j)=fobj([x(i),y(j),0,0]);
        end          
    end
end

surfc(x,y,f,'LineStyle','none');

end


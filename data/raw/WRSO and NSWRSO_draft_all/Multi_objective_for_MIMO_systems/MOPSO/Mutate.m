%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA121
% Project Title: Multi-Objective Particle Swarm Optimization (MOPSO)
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%
function xnew=Mutate(x,pm,VarMin,VarMax)
    VarMax = max(VarMax);
    VarMin = min(VarMin);
    nVar=numel(x);
    j=randi([2 nVar]);
    dx=pm*(VarMax-VarMin);
    
    lb=x(j)-dx;
    if lb<VarMin
        lb=VarMin;
    end
    
    ub=x(j)+dx;
    if ub>VarMax
        ub=VarMax;
    end
  
    xnew=x;
    xnew(j)=unifrnd(lb,ub);
    xnew(1)= 0;
end
function xnew=Mutatetf1(x,pm,VarMin,VarMax)
    nVar=numel(x);
    j=randi([1 nVar]);
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
    xnew(j)=lb(j)-((lb(j)-ub(j))*rand);
end
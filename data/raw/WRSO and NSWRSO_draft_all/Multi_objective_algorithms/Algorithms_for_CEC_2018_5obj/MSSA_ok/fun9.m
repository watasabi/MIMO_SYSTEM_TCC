function y = fun9(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
D = MultiObj.nVar;
Points = [];
[thera,rho] = cart2pol(0,1);
[Points(:,1),Points(:,2)] = pol2cart(thera-(1:M)*2*pi/M,rho);
% Feasible polygon
Points = [];
[thera,rho] = cart2pol(0,1);
[Points(:,1),Points(:,2)] = pol2cart(thera-(1:M)*2*pi/M,rho);
% Infeasible polygons
head = repmat((1:M)',ceil(M/2-2),1);
tail = repmat(1:ceil(M/2-2),M,1);
tail = head + tail(:);
Polygons = cell(1,length(head));
for i = 1:length(Polygons)
    Polygons{i} = Points(mod((head(i):tail(i))-1,M)+1,:);
    Polygons{i} = [Polygons{i};repmat(2*Intersection(Points(mod([head(i)-1,head(i),tail(i),tail(i)+1]-1,M)+1,:)),size(Polygons{i},1),1)-Polygons{i}];
end
Infeasible = getInfeasible(x,Polygons,Points);
while any(Infeasible)
    x(Infeasible,:) = rand(sum(Infeasible),D).*repmat(MultiObj.var_max-MultiObj.var_min,sum(Infeasible),1) + repmat(MultiObj.var_min,sum(Infeasible),1);
    Infeasible = getInfeasible(x,Polygons,Points);
end

y = zeros(size(x,1),size(Points,1));
for m = 1 : size(Points,1)
    y(:,m) = Point2Line(x,Points(mod(m-1:m,size(Points,1))+1,:));
end
end

function r = Intersection(p)
    if p(1,1) == p(2,1)
        r(1) = p(1,1);
        r(2) = p(3,2)+(r(1)-p(3,1))*(p(3,2)-p(4,2))/(p(3,1)-p(4,1));
    elseif p(3,1) == p(4,1)
        r(1) = p(3,1);
        r(2) = p(1,2)+(r(1)-p(1,1))*(p(1,2)-p(2,2))/(p(1,1)-p(2,1));
    else
        k1   = (p(1,2)-p(2,2))/(p(1,1)-p(2,1));
        k2   = (p(3,2)-p(4,2))/(p(3,1)-p(4,1));
        r(1) = (k1*p(1,1)-k2*p(3,1)+p(3,2)-p(1,2))/(k1-k2);
        r(2) = p(1,2)+(r(1)-p(1,1))*k1;
    end
end

function Infeasible = getInfeasible(x,Polygons,Points)
    Infeasible = false(size(x,1),1);
    for i = 1 : length(Polygons)
        Infeasible = Infeasible | inpolygon(x(:,1),x(:,2),Polygons{i}(:,1),Polygons{i}(:,2));
    end
    Infeasible = Infeasible & ~inpolygon(x(:,1),x(:,2),Points(:,1),Points(:,2));
end

function Distance = Point2Line(x,Line)
    Distance = abs((Line(1,1)-x(:,1)).*(Line(2,2)-x(:,2))-(Line(2,1)-x(:,1)).*(Line(1,2)-x(:,2)))./sqrt((Line(1,1)-Line(2,1)).^2+(Line(1,2)-Line(2,2)).^2);
end
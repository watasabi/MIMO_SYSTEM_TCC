function y = fun5(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
x(:,1:M-1) = x(:,1:M-1).^100;
g = sum((x(:,M:end)-0.5).^2,2);
a = fliplr(cumprod([ones(size(g,1),1),cos(x(:,1:M-1)*pi/2)],2));
b = [ones(size(g,1),1),sin(x(:,M-1:-1:1)*pi/2)];
aux = repmat(1+g,1,M).*a.*b;
y = aux.*repmat(2.^(M:-1:1),size(g,1),1);
end
function y = fun3(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
[N,D] = size(x);
g = 100*(D-M+1+sum((x(:,M:end)-0.5).^2-cos(20.*pi.*(x(:,M:end)-0.5)),2));
a = fliplr(cumprod([ones(size(g,1),1),cos(x(:,1:M-1)*pi/2)],2));
b = [ones(size(g,1),1),sin(x(:,M-1:-1:1)*pi/2)];
aux = repmat(1+g,1,M).*a.*b;
y = [aux(:,1:M-1).^4,aux(:,M).^2];
end

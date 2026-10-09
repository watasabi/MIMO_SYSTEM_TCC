function y = fun4(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
[N,D] = size(x);
g = 100*(D-M+1+sum((x(:,M:end)-0.5).^2-cos(20.*pi.*(x(:,M:end)-0.5)),2));
a = fliplr(cumprod([ones(N,1),cos(x(:,1:M-1)*pi/2)],2));
b = [ones(N,1),sin(x(:,M-1:-1:1)*pi/2)];
aux = repmat(1+g,1,M) - repmat(1+g,1,M).*a.*b;
y = aux.*repmat(2.^(1:M),N,1);
end
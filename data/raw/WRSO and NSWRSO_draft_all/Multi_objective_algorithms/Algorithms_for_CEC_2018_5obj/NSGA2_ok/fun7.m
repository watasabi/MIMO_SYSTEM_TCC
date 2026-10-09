function y = fun7(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
y = zeros(size(x,1),M);
g = 1+9*mean(x(:,M:end),2);
y(:,1:M-1) = x(:,1:M-1);
y(:,M) = (1+g).*(M-sum(y(:,1:M-1)./(1+repmat(g,1,M-1)).*(1+sin(3*pi.*y(:,1:M-1))),2));
end
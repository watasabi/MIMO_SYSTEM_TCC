function y = fun1(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
g = sum((x(:,M:end)-0.5).^2,2);
a = fliplr(cumprod([ones(size(g,1),1),x(:,1:M-1)],2));
b = [ones(size(g,1),1),1-x(:,M-1:-1:1)];
y = repmat(1+g,1,M) - repmat(1+g,1,M).*a.*b;
end

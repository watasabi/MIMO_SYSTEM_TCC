function y = fun6(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
I = 2;
g = sum((x(:,M:end)-0.5).^2,2);
Temp = repmat(g,1,M-I);
x(:,I:M-1) = (1+2*Temp.*x(:,I:M-1))./(2+2*Temp);
a = fliplr(cumprod([ones(size(g,1),1),cos(x(:,1:M-1)*pi/2)],2));
b = [ones(size(g,1),1),sin(x(:,M-1:-1:1)*pi/2)];
y = repmat(1+100*g,1,M).*a.*b;
end
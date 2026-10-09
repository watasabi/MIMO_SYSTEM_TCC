function y = fun13(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
[N,D] = size(x);
Y = x - 2*repmat(x(:,2),1,D).*sin(2*pi*repmat(x(:,1),1,D)+repmat(1:D,N,1)*pi/D);
y(:,1) = sin(x(:,1)*pi/2) + 2*mean(Y(:,4:3:D).^2,2);
y(:,2) = cos(x(:,1)*pi/2).*sin(x(:,2)*pi/2) + 2*mean(Y(:,5:3:D).^2,2);
y(:,3) = cos(x(:,1)*pi/2).*cos(x(:,2)*pi/2) + 2*mean(Y(:,3:3:D).^2,2);
y(:,4:M) = repmat(y(:,1).^2+y(:,2).^10+y(:,3).^10+2*mean(Y(:,4:D).^2,2),1,M-3);
end

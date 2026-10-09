function y = fun15(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
nVar = M*20;
nk = 2;
c = 3.8*0.1*(1-0.1);
for i = 1:M-1
    c = [c,3.8.*c(end).*(1-c(end))];
end
sublen = ceil(round(c./sum(c).*nVar)/nk);
len    = [0,cumsum(sublen*nk)];
[N,D] = size(x);
x(:,M:D) = (1+repmat(cos((M:D)./D*pi/2),N,1)).*x(:,M:D) - repmat(x(:,1)*10,1,D-M+1);
G = zeros(N,M);
for i = 1:2:M
    for j = 1:nk
        G(:,i) = G(:,i) + Griewank(x(:,len(i)+M-1+(j-1)*sublen(i)+1:len(i)+M-1+j*sublen(i)));
    end
end
for i = 2:2:M
    for j = 1:nk
        G(:,i) = G(:,i) + Sphere(x(:,len(i)+M-1+(j-1)*sublen(i)+1:len(i)+M-1+j*sublen(i)));
    end
end
G = G./repmat(sublen,N,1)./nk;
y = (1+G+[G(:,2:end),zeros(N,1)]).*(1-fliplr(cumprod([ones(N,1),cos(x(:,1:M-1)*pi/2)],2)).*[ones(N,1),sin(x(:,M-1:-1:1)*pi/2)]);
end

function f = Griewank(x1)
    f = sum(x1.^2,2)./4000 - prod(cos(x1./repmat(sqrt(1:size(x1,2)),size(x1,1),1)),2) + 1;
end

function f = Sphere(x2)
    f = sum(x2.^2,2);
end
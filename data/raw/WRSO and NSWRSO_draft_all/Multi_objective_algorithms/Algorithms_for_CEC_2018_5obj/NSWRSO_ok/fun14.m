function y = fun14(x,MultiObj)
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
x(:,M:D) = (1+repmat((M:D)./D,N,1)).*x(:,M:D) - repmat(x(:,1)*10,1,D-M+1);
G = zeros(N,M);
for i = 1:2:M
    for j = 1:nk
        G(:,i) = G(:,i) + Rastrigin(x(:,len(i)+M-1+(j-1)*sublen(i)+1:len(i)+M-1+j*sublen(i)));
    end
end
for i = 2:2:M
    for j = 1:nk
        G(:,i) = G(:,i) + Rosenbrock(x(:,len(i)+M-1+(j-1)*sublen(i)+1:len(i)+M-1+j*sublen(i)));
    end
end
G = G./repmat(sublen,N,1)./nk;
y = (1+G).*fliplr(cumprod([ones(N,1),x(:,1:M-1)],2)).*[ones(N,1),1-x(:,M-1:-1:1)];
end

function f = Rastrigin(x1)
    f = sum(x1.^2-10.*cos(2.*pi.*x1)+10,2);
end

function f = Rosenbrock(x2)
    f = sum(100.*(x2(:,1:size(x2,2)-1).^2-x2(:,2:size(x2,2))).^2+(x2(:,1:size(x2,2)-1)-1).^2,2);
end
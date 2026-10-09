function y = fun10(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
K = M - 1;
[N,D] = size(x);
L = D - K;
D = 1;
S = 2:2:2*M;
A = ones(1,M-1);
z01 = x./repmat(2:2:size(x,2)*2,N,1);
t1 = zeros(N,K+L);
t1(:,1:K) = z01(:,1:K);
t1(:,K+1:end) = s_linear(z01(:,K+1:end),0.35);
t2 = zeros(N,K+L);
t2(:,1:K) = t1(:,1:K);
t2(:,K+1:end) = b_flat(t1(:,K+1:end),0.8,0.75,0.85);
t3 = zeros(N,K+L);
t3 = b_poly(t2,0.02);
t4 = zeros(N,M);
for i = 1:M-1
    t4(:,i) = r_sum(t3(:,(i-1)*K/(M-1)+1:i*K/(M-1)),2*((i-1)*K/(M-1)+1):2:2*i*K/(M-1));
end
t4(:,M) = r_sum(t3(:,K+1:K+L),2*(K+1):2:2*(K+L));
xx = zeros(N,M);
for i = 1 : M-1
    xx(:,i) = max(t4(:,M),A(i)).*(t4(:,i)-0.5)+0.5;
end
xx(:,M) = t4(:,M);
h = convex(xx);
h(:,M) = mixed(xx);
y = repmat(D*xx(:,M),1,M) + repmat(S,N,1).*h;
end

function Output = s_linear(y1,A)
    Output = abs(y1-A)./abs(floor(A-y1)+A);
end

function Output = b_flat(y2,A,B,C)
    Output = A+min(0,floor(y2-B))*A.*(B-y2)/B-min(0,floor(C-y2))*(1-A).*(y2-C)/(1-C);
    Output = roundn(Output,-6);
end

function Output = b_poly(y3,a)
    Output = y3.^a;
end

function Output = r_sum(y4,w)
    Output = sum(y4.*repmat(w,size(y4,1),1),2)./sum(w);
end

function Output = convex(x1)
    Output = fliplr(cumprod([ones(size(x1,1),1),1-cos(x1(:,1:end-1)*pi/2)],2)).*[ones(size(x1,1),1),1-sin(x1(:,end-1:-1:1)*pi/2)];
end

function Output = mixed(x2)
    Output = 1-x2(:,1)-cos(10*pi*x2(:,1)+pi/2)/10/pi;
end
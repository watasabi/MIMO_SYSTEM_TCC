function y = fun11(x,MultiObj)
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
t2 = zeros(N,K+L/2);
t2(:,1:K) = t1(:,1:K);
t2(:,K+1:K+L/2) = (t1(:,K+1:2:end) + t1(:,K+2:2:end) + 2*abs(t1(:,K+1:2:end)-t1(:,K+2:2:end)))/3;
t3 = zeros(N,M);
for i = 1:M-1
    t3(:,i) = r_sum(t2(:,(i-1)*K/(M-1)+1:i*K/(M-1)),ones(1,K/(M-1)));
end
t3(:,M) = r_sum(t2(:,K+1:K+L/2),ones(1,L/2));
xx = zeros(N,M);
for i = 1:M-1
    xx(:,i) = max(t3(:,M),A(i)).*(t3(:,i)-0.5)+0.5;
end
xx(:,M) = t3(:,M);
h = convex(xx);
h(:,M) = disc(xx);
y = repmat(D*xx(:,M),1,M) + repmat(S,N,1).*h;
end

function Output = s_linear(y1,A)
    Output = abs(y1-A)./abs(floor(A-y1)+A);
end

function Output = r_nonsep(y2,A)
    Output = zeros(size(y2,1),1);
    for j = 1:size(y2,2)
        Temp = zeros(size(y2,1),1);
        for k = 0 : A-2
            Temp = Temp+abs(y2(:,j)-y2(:,1+mod(j+k,size(y2,2))));
        end
        Output = Output+y2(:,j)+Temp;
    end
    Output = Output./(size(y2,2)/A)/ceil(A/2)/(1+2*A-2*ceil(A/2));
end

function Output = r_sum(y3,w)
    Output = sum(y3.*repmat(w,size(y3,1),1),2)./sum(w);
end

function Output = convex(x1)
    Output = fliplr(cumprod([ones(size(x1,1),1),1-cos(x1(:,1:end-1)*pi/2)],2)).*[ones(size(x1,1),1),1-sin(x1(:,end-1:-1:1)*pi/2)];
end

function Output = disc(x2)
    Output = 1-x2(:,1).*(cos(5*pi*x2(:,1))).^2;
end

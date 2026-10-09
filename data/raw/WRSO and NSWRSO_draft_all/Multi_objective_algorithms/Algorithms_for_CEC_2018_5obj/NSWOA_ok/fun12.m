function y = fun12(x,MultiObj)
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
Y = (fliplr(cumsum(fliplr(z01),2))-z01)./repmat(K+L-1:-1:0,N,1);
t1(:,1:K+L-1) = z01(:,1:K+L-1).^(0.02+(50-0.02)*(0.98/49.98-(1-2*Y(:,1:K+L-1)).*abs(floor(0.5-Y(:,1:K+L-1))+0.98/49.98)));
t1(:,end) = z01(:,end);
t2 = zeros(N,K+L);
t2(:,1:K) = s_decept(t1(:,1:K),0.35,0.001,0.05);
t2(:,K+1:end) = s_multi(t1(:,K+1:end),30,95,0.35);
t3 = zeros(N,M);
for i = 1:M-1
    t3(:,i) = r_nonsep(t2(:,(i-1)*K/(M-1)+1:i*K/(M-1)),K/(M-1));
end
SUM = zeros(N,1);
for i = K+1:K+L-1
    for j = i+1 : K+L
        SUM = SUM + abs(t2(:,i)-t2(:,j));
    end
end
t3(:,M) = (sum(t2(:,K+1:end),2)+SUM*2)/ceil(L/2)/(1+2*L-2*ceil(L/2));
xx = zeros(N,M);
for i = 1:M-1
    xx(:,i) = max(t3(:,M),A(i)).*(t3(:,i)-0.5)+0.5;
end
xx(:,M) = t3(:,M);
h = concave(xx);
y = repmat(D*xx(:,M),1,M) + repmat(S,N,1).*h;
end

function Output = b_param(y1,Y,A,B,C)
    Output = y1.^(B+(C-B)*(A-(1-2*Y).*abs(floor(0.5-Y)+A)));
end

function Output = r_sum(y2,w)
    Output = sum(y2.*repmat(w,size(y2,1),1),2)./sum(w);
end

function Output = s_decept(y3,A,B,C)
    Output = 1+(abs(y3-A)-B).*(floor(y3-A+B)*(1-C+(A-B)/B)/(A-B)+floor(A+B-y3)*(1-C+(1-A-B)/B)/(1-A-B)+1/B);
end

function Output = s_multi(y4,A,B,C)
    Output = (1+cos((4*A+2)*pi*(0.5-abs(y4-C)/2./(floor(C-y4)+C)))+4*B*(abs(y4-C)/2./(floor(C-y4)+C)).^2)/(B+2);
end

function Output = r_nonsep(y5,A)
    Output = zeros(size(y5,1),1);
    for j = 1 : size(y5,2)
        Temp = zeros(size(y5,1),1);
        for k = 0:A-2
            Temp = Temp+abs(y5(:,j)-y5(:,1+mod(j+k,size(y5,2))));
        end
        Output = Output+y5(:,j)+Temp;
    end
    Output = Output./(size(y5,2)/A)/ceil(A/2)/(1+2*A-2*ceil(A/2));
end

function Output = concave(x1)
    Output = fliplr(cumprod([ones(size(x1,1),1),sin(x1(:,1:end-1)*pi/2)],2)).*[ones(size(x1,1),1),cos(x1(:,end-1:-1:1)*pi/2)];
end
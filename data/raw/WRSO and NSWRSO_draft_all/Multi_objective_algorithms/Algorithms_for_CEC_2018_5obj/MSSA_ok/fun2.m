function y = fun2(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
[N,D] = size(x);
g = zeros(N,M);
    for m = 1:M
        if m < M
            g(:,m) = sum(((x(:,M+(m-1)*floor((D-M+1)/M):M+m*floor((D-M+1)/M)-1)/2+1/4)-0.5).^2,2);
        else
            g(:,m) = sum(((x(:,M+(M-1)*floor((D-M+1)/M):D)/2+1/4)-0.5).^2,2);
        end
    end
    a = fliplr(cumprod([ones(size(g,1),1),cos((x(:,1:M-1)/2+1/4)*pi/2)],2));
    b = [ones(size(g,1),1),sin((x(:,M-1:-1:1)/2+1/4)*pi/2)];
    y = (1+g).*(a.*b);
end

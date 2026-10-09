function y = fun8(x,MultiObj)
% Objective function : Test problem 'DTLZ2'.
%*************************************************************************
M = MultiObj.M;
Points = [];
[thera,rho] = cart2pol(0,1);
[Points(:,1),Points(:,2)] = pol2cart(thera-(1:M)*2*pi/M,rho);
y = pdist2(x,Points);
end
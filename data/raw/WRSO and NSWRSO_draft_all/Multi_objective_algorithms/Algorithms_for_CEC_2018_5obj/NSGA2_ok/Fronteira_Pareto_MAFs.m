close all
clear all
clc

M          = 3;     % Número de objetivos
input      = 10000; % Número de pontos da PF

%--------------------------------------------------------------------------
    % This problem is inverted DTLZ1 with the g function of DTLZ2
%--------------------------------------------------------------------------    
            f = UniformPoint(input,M);
            PFMaF1 = 1-f;
            if M == 3
                figure
                plot3(PFMaF1(:,1),PFMaF1(:,2),PFMaF1(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF1
%--------------------------------------------------------------------------
    % This problem is DTLZ2BZ
%--------------------------------------------------------------------------
            f = max(UniformPoint(input,M),1e-6);
            c = zeros(size(f,1),M-1);
            for i = 1 : size(f,1)
                for j = 2 : M
                    temp = f(i,j)/f(i,1)*prod(c(i,M-j+2:M-1));
                    c(i,M-j+1) = sqrt(1/(1+temp^2));
                end
            end
            if M > 5
                c = c.*(cos(pi/8)-cos(3*pi/8)) + cos(3*pi/8);
            else
                c(any(c<cos(3*pi/8)|c>cos(pi/8),2),:) = [];
            end
            f = fliplr(cumprod([ones(size(c,1),1),c(:,1:M-1)],2)).*[ones(size(c,1),1),sqrt(1-c(:,M-1:-1:1).^2)];
            PFMaF2 = f;
            if M == 3
                figure
                plot3(PFMaF2(:,1),PFMaF2(:,2),PFMaF2(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF2
%--------------------------------------------------------------------------
    % This problem is convex DTLZ3
%--------------------------------------------------------------------------
            f    = UniformPoint(input,M).^2;
            temp = sum(sqrt(f(:,1:end-1)),2) + f(:,end);
            f    = f./[repmat(temp.^2,1,size(f,2)-1),temp];
            PFMaF3 = f;
            if M == 3
                figure
                plot3(PFMaF3(:,1),PFMaF3(:,2),PFMaF3(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF3
%--------------------------------------------------------------------------
    % This problem is inverted badly-scaled DTLZ3
%--------------------------------------------------------------------------
            f = UniformPoint(input,M);
            f = f./repmat(sqrt(sum(f.^2,2)),1,M);
            f = (1-f).*repmat(2.^(1:M),size(f,1),1);
            PFMaF4 = f;
            if M == 3
                figure
                plot3(PFMaF4(:,1),PFMaF4(:,2),PFMaF4(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF4
%--------------------------------------------------------------------------
    % This problem is scaled DTLZ4
%--------------------------------------------------------------------------
            f = UniformPoint(input,M);
            f = f./repmat(sqrt(sum(f.^2,2)),1,M);
            f = f.*repmat(2.^(M:-1:1),size(f,1),1);
            PFMaF5 = f;
            if M == 3
                figure
                plot3(PFMaF5(:,1),PFMaF5(:,2),PFMaF5(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF5
%--------------------------------------------------------------------------
    % This problem is DTLZ5(I,M) with I=2
%--------------------------------------------------------------------------
            I = 2;
            f = UniformPoint(input,I);
            f = f./repmat(sqrt(sum(f.^2,2)),1,size(f,2));
            f = [f(:,ones(1,M-size(f,2))),f];
            f = f./sqrt(2).^repmat(max([M-I,M-I:-1:2-I],0),size(f,1),1);
            PFMaF6 = f;
            if M == 3
                figure
                plot3(PFMaF6(:,1),PFMaF6(:,2),PFMaF6(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF6
%--------------------------------------------------------------------------
    % This problem is DTLZ7
%--------------------------------------------------------------------------
            interval     = [0,0.251412,0.631627,0.859401];
            median       = (interval(2)-interval(1))/(interval(4)-interval(3)+interval(2)-interval(1));
            X            = ReplicatePoint(input,M-1);
            X(X<=median) = X(X<=median)*(interval(2)-interval(1))/median+interval(1);
            X(X>median)  = (X(X>median)-median)*(interval(4)-interval(3))/(1-median)+interval(3);
            f            = [X,2*(M-sum(X/2.*(1+sin(3*pi.*X)),2))];
            PFMaF7 = f;
            if M == 3
                figure
                plot3(PFMaF7(:,1),PFMaF7(:,2),PFMaF7(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF7
%--------------------------------------------------------------------------
    % This problem is multi-point distance minimization problem (similar to
    % Pareto-box problem but more difficult)
%--------------------------------------------------------------------------
            [thera,rho] = cart2pol(0,1);
            [Points(:,1),Points(:,2)] = pol2cart(thera-(1:M)*2*pi/M,rho);
            [X,Y]     = ndgrid(linspace(-1,1,ceil(sqrt(input))));
            ND        = inpolygon(X(:),Y(:),Points(:,1),Points(:,2));
            PopObj    = pdist2([X(ND),Y(ND)],Points);
            PFMaF8 = PopObj;
            figure
            cla
            plot(Points([1:end,1],1),Points([1:end,1],2),'-k','LineWidth',1.5);
            hold
            plot(Points(:,1),Points(:,2),'ok','MarkerSize',6,'Marker','o','Markerfacecolor',[1 1 1],'Markeredgecolor',[.4 .4 .4]);
            xlabel('\itx\rm_1'); ylabel('\itx\rm_2');
            plot(X(ND),Y(ND),'o')
            xlabel('x1')
            ylabel('x2')
            save PFMaF8
%--------------------------------------------------------------------------
    % This problem is multi-line distance minimization problem
%--------------------------------------------------------------------------
            % Feasible polygon
            [thera,rho] = cart2pol(0,1);
            [Points(:,1),Points(:,2)] = pol2cart(thera-(1:M)*2*pi/M,rho);
            % Infeasible polygons
            head     = repmat((1:M)',ceil(M/2-2),1);
            tail     = repmat(1:ceil(M/2-2),M,1);
            tail     = head + tail(:);
            Polygons = cell(1,length(head));
            for i = 1 : length(Polygons)
                Polygons{i} = Points(mod((head(i):tail(i))-1,M)+1,:);
                Polygons{i} = [Polygons{i};repmat(2*Intersection(Points(mod([head(i)-1,head(i),tail(i),tail(i)+1]-1,M)+1,:)),size(Polygons{i},1),1)-Polygons{i}];
            end
             [X,Y]      = ndgrid(linspace(-1,1,ceil(sqrt(input))));
             ND         = inpolygon(X(:),Y(:),Points(:,1),Points(:,2));
             PopDec     = [X(ND),Y(ND)];
             Infeasible = getInfeasible(PopDec,Polygons,Points);
             D = 2;
             lower      = [-10000,-10000];
             upper      = [10000,10000];
             while any(Infeasible)
                PopDec(Infeasible,:) = rand(sum(Infeasible),D).*repmat(upper-lower,sum(Infeasible),1) + repmat(lower,sum(Infeasible),1);
                Infeasible           = getInfeasible(PopDec,Polygons,Points);
            end
             PopObj = zeros(size(PopDec,1),size(Points,1));
            for m = 1 : size(Points,1)
                PopObj(:,m) = Point2Line(PopDec,Points(mod(m-1:m,size(Points,1))+1,:));
            end
            PFMaF9 = PopObj;
%             [~,PopObj] = MaF9('value',Global,[X(ND),Y(ND)]);
%             varargout  = {PopObj};
            figure
            cla;
            plot(Points([1:end,1],1),Points([1:end,1],2),'-k','LineWidth',1.5);
            hold
            plot(Points(:,1),Points(:,2),'ok','MarkerSize',6,'Marker','o','Markerfacecolor',[1 1 1],'Markeredgecolor',[.4 .4 .4]);
            xlabel('\itx\rm_1'); ylabel('\itx\rm_2');
            plot(X(ND),Y(ND),'o')
            xlabel('x1')
            ylabel('x2')
            save PFMaF9
%--------------------------------------------------------------------------
    % This problem is WFG1 with K=M-1
%--------------------------------------------------------------------------
            h = UniformPoint(input,M);
            for i = 1 : size(h,1)
                c = ones(1,M);
                k = find(h(i,:)~=0,1);
                for j = k+1 : M
                    temp     = h(i,j)/h(i,k)*prod(1-c(M-j+2:M-k));
                    c(M-j+1) = (temp^2-temp+sqrt(2*temp))/(temp^2+1);
                end
                for j = 1 : M
                    h(i,j) = prod(1-c(1:M-j)).*(1-sqrt(1-c(M-j+1)^2));
                end
                temp   = acos(c(1))*2/pi;                   
                h(i,M) = 1 - temp - cos(10*pi*temp+pi/2)/10/pi;
            end
            h = repmat(2:2:2*M,size(h,1),1).*h;
            PFMaF10 = h;
            if M == 3
                figure
                plot3(PFMaF10(:,1),PFMaF10(:,2),PFMaF10(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF10
%--------------------------------------------------------------------------
    % This problem is WFG2 with K=M-1
%--------------------------------------------------------------------------
            x = ReplicatePoint(input,M-1);
            x(:,1) = sqrt(x(:,1));
            x = [x,zeros(size(x,1),1)];
            h = convex(x);
            h(:,M) = disc(x);
            h = repmat(2:2:2*M,size(h,1),1).*h;
            PFMaF11 = h(NDSort(h,1)==1,:);
            if M == 3
                figure
                plot3(PFMaF11(:,1),PFMaF11(:,2),PFMaF11(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF11
%--------------------------------------------------------------------------
    % This problem is WFG9 with K=M-1
%--------------------------------------------------------------------------
            h = UniformPoint(input,M);
            h = h./repmat(sqrt(sum(h.^2,2)),1,M);
            h = repmat(2:2:2*M,size(h,1),1).*h;
            PFMaF12 = h;
            if M == 3
                figure
                plot3(PFMaF12(:,1),PFMaF12(:,2),PFMaF12(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF12
%--------------------------------------------------------------------------
    % This problem is P7
%--------------------------------------------------------------------------
            f = UniformPoint(input,3);
            f = f./repmat(sqrt(sum(f.^2,2)),1,3);
            f = [f,repmat(f(:,1).^2+f(:,2).^10+f(:,3).^10,1,M-3)];
            PFMaF13 = f;
            if M == 3
                figure
                plot3(PFMaF13(:,1),PFMaF13(:,2),PFMaF13(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF13
%--------------------------------------------------------------------------
    % This problem is LSMOP3 with nk=2
%--------------------------------------------------------------------------
            f = UniformPoint(input,M);
            PFMaF14 = f;
            if M == 3
                figure
                plot3(PFMaF14(:,1),PFMaF14(:,2),PFMaF14(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF14
%--------------------------------------------------------------------------
    % This problem is inverted LSMOP8 with nk=2
%--------------------------------------------------------------------------
            f = UniformPoint(input,M);
            f = f./repmat(sqrt(sum(f.^2,2)),1,M);
            PFMaF15 = 1-f;
            if M == 3
                figure
                plot3(PFMaF15(:,1),PFMaF15(:,2),PFMaF15(:,3),'o')
                xlabel('f1')
                ylabel('f2')
                zlabel('f3')
            end
            save PFMaF15
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function [W,N] = UniformPoint(N,M)
    H1 = 1;
    while nchoosek(H1+M,M-1) <= N
        H1 = H1 + 1;
    end
    W = nchoosek(1:H1+M-1,M-1) - repmat(0:M-2,nchoosek(H1+M-1,M-1),1) - 1;
    W = ([W,zeros(size(W,1),1)+H1]-[zeros(size(W,1),1),W])/H1;
    if H1 < M
        H2 = 0;
        while nchoosek(H1+M-1,M-1)+nchoosek(H2+M,M-1) <= N
            H2 = H2 + 1;
        end
        if H2 > 0
            W2 = nchoosek(1:H2+M-1,M-1) - repmat(0:M-2,nchoosek(H2+M-1,M-1),1) - 1;
            W2 = ([W2,zeros(size(W2,1),1)+H2]-[zeros(size(W2,1),1),W2])/H2;
            W  = [W;W2/2+1/(2*M)];
        end
    end
    W = max(W,1e-6);
    N = size(W,1);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function [FrontNo,MaxFNo] = NDSort(varargin)
    if nargin == 2
        PopObj = varargin{1};
        nSort  = varargin{2};
    else
        PopObj = varargin{1};
        PopCon = varargin{2};
        nSort  = varargin{3};
        Infeasible           = any(PopCon>0,2);
        PopObj(Infeasible,:) = repmat(max(PopObj,[],1),sum(Infeasible),1) + repmat(sum(max(0,PopCon(Infeasible,:)),2),1,size(PopObj,2));
    end
    [PopObj,~,Loc] = unique(PopObj,'rows');
    Table          = hist(Loc,1:max(Loc));
    [N,M]          = size(PopObj);
    [PopObj,rank]  = sortrows(PopObj);
    FrontNo        = inf(1,N);
    MaxFNo         = 0;
    while sum(Table(FrontNo<inf)) < min(nSort,length(Loc))
        MaxFNo = MaxFNo + 1;
        for i = 1 : N
            if FrontNo(i) == inf
                Dominated = false;
                for j = i-1 : -1 : 1
                    if FrontNo(j) == MaxFNo
                        m = 2;
                        while m <= M && PopObj(i,m) >= PopObj(j,m)
                            m = m + 1;
                        end
                        Dominated = m > M;
                        if Dominated || M == 2
                            break;
                        end
                    end
                end
                if ~Dominated
                    FrontNo(i) = MaxFNo;
                end
            end
        end
    end
    FrontNo(rank) = FrontNo;
    FrontNo       = FrontNo(Loc);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function W = ReplicatePoint(SampleNum,M)
    if M > 1
        SampleNum = (ceil(SampleNum^(1/M)))^M;
        Gap       = 0:1/(SampleNum^(1/M)-1):1;
        eval(sprintf('[%s]=ndgrid(Gap);',sprintf('c%d,',1:M)))
        eval(sprintf('W=[%s];',sprintf('c%d(:),',1:M)))
    else
        W = (0:1/(SampleNum-1):1)';
    end
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function r = Intersection(p)
    if p(1,1) == p(2,1)
        r(1) = p(1,1);
        r(2) = p(3,2)+(r(1)-p(3,1))*(p(3,2)-p(4,2))/(p(3,1)-p(4,1));
    elseif p(3,1) == p(4,1)
        r(1) = p(3,1);
        r(2) = p(1,2)+(r(1)-p(1,1))*(p(1,2)-p(2,2))/(p(1,1)-p(2,1));
    else
        k1   = (p(1,2)-p(2,2))/(p(1,1)-p(2,1));
        k2   = (p(3,2)-p(4,2))/(p(3,1)-p(4,1));
        r(1) = (k1*p(1,1)-k2*p(3,1)+p(3,2)-p(1,2))/(k1-k2);
        r(2) = p(1,2)+(r(1)-p(1,1))*k1;
    end
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function Infeasible = getInfeasible(PopDec,Polygons,Points)
    Infeasible = false(size(PopDec,1),1);
    for i = 1 : length(Polygons)
        Infeasible = Infeasible | inpolygon(PopDec(:,1),PopDec(:,2),Polygons{i}(:,1),Polygons{i}(:,2));
    end
    Infeasible = Infeasible & ~inpolygon(PopDec(:,1),PopDec(:,2),Points(:,1),Points(:,2));
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function Distance = Point2Line(PopDec,Line)
    Distance = abs((Line(1,1)-PopDec(:,1)).*(Line(2,2)-PopDec(:,2))-(Line(2,1)-PopDec(:,1)).*(Line(1,2)-PopDec(:,2)))./sqrt((Line(1,1)-Line(2,1)).^2+(Line(1,2)-Line(2,2)).^2);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function Output = convex(x)
    Output = fliplr(cumprod([ones(size(x,1),1),1-cos(x(:,1:end-1)*pi/2)],2)).*[ones(size(x,1),1),1-sin(x(:,end-1:-1:1)*pi/2)];
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function Output = disc(x)
    Output = 1-x(:,1).*(cos(5*pi*x(:,1))).^2;
end
%--------------------------------------------------------------------------
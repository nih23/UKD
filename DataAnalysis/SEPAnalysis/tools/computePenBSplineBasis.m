function [ Z,D,Ds ] = computePenBSplineBasis(  m, p, q, noPoints, addIdx )
%PENSPLINE Penalized Spline Smoothing using min(m/4,40) knots and
%polynomial basis function of order 2
% m: no of datapoints
% p: polynomial degree
% q: degree of penalty

if(~exist('noPoints','var'))
    k = min(round(m/4),40); % Ruppert, 2002
else
    k = noPoints;
end
%p = 2;

k = k;

T = (1:m);

X = [ones(1,m)'];

Z = [];
%idx = (linspace(T(1),T(end),k));
%t = (linspace(T(1),T(end),k+1));
t = round(([1:k] ./ (k+1)) .* m); % t_k -> kth sample quantile
t = [1*ones(1,p) t m*ones(1,p)];
if(exist('addIdx','var'))
   t = [t addIdx];
   k = k + length(addIdx);
   t = sort(t);  
end
for i=1:p
    X = [X (T.^i)'];
end




% bspline degree zero
for i=1:length(t)-1
    [~,k_j] = min(abs(T - t(i)));
    [~,k_jPlusOne] = min(abs(T - t(i+1)));
    Bj0(1:m) = 0;
    if(k_j < m && k_jPlusOne == m)
        Bj0(k_jPlusOne) = 1;
    end
    Bj0(k_j : k_jPlusOne-1) = 1;
    Z = [Z; Bj0];
end


B{1} = Z;

for d = 2 : p
    Z = [];
    for i=1:k+2*p - d
        Z_pjMinusOne = B{d-1};
        
        ti = t((i));
        tid = t(i+d-1);
        ti1 = t((i+1));
        ti1d = t(i+d);
        
        Bj(1:m) = 0;
        
        if( (tid - ti) > 0)
           Bj = ((T - ti) ./ (tid - ti)) .* Z_pjMinusOne(i,:); 
        end
        
        if( (ti1d - ti1) > 0)
            Bj = Bj + ((ti1d - T) ./ (ti1d - ti1)) .* Z_pjMinusOne(i+1,:);
        end
        Z = [Z; Bj];
    end
    B{d} = Z;
    
end

Z = Z';
[n,m] = size(Z);
D=diff(eye(m),q)'*diff(eye(m),q);
Ds = diff(eye(m),q);
end
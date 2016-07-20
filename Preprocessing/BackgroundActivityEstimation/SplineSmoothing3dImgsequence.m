function [ RdMin, RSS ] = SplineSmoothing3dImgsequence( seq, seq_timing,ddx,ddy,ddt,oldNUCModel )
[n,dt] = size(seq);

fprintf('Penalty-free 3D Spline Smoothing\n');
[shutterPtnrV3,~,t_S] = computeShutterPatternV3(seq,seq_timing);

%seq = double(seq);
seq = reshape(seq,[480,640,dt]);

[B1,Dx] = computePenBSplineBasis(480,3,2,ddx); % 10 used for evaluation working  % 35
[B2,Dy] = computePenBSplineBasis(640,3,2,ddy);                                   % 35

if(exist('oldNUCModel','var'))
    fprintf('> old NUC model\n');
    [Bdrift,Draw] = computePenBSplineBasis(dt,3,2,ddt); % 350 ; 150                 % 50    
    BasisTimedomain = [Bdrift shutterPtnrV3'];   
else
    fprintf('> new NUC correction\n');
    [Bdrift,Draw] = computePenBSplineBasis(dt,3,2,ddt,t_S); % 350 ; 150                 % 50    
    BasisTimedomain = Bdrift;
end
%[Bdrift,Draw] = computePenBSplineBasis(dt,3,2,ddt); % 350 ; 150                 % 50
%[Bdrift,Draw] = computePenBSplineBasis(dt,3,2,ddt,t_S);
%BasisTimedomain = [Bdrift shutterPtnrV3'];
%BasisTimedomain = Bdrift;
[~,n1] = size(B1);
[~,n2] = size(B2);
[~,n3] = size(BasisTimedomain);
%[~,n3BSpline] = size(Bdrift);

B1 = ndSparse(B1);
B2 = ndSparse(B2);
BasisTimedomain = ndSparse(BasisTimedomain);

W = ndSparse(ones(480,640,dt));
BB = ndSparse(box(B1,B1));
F = (rho(BB,W,1));
F=ndSparse(F);
clear BB;
clear W;

BB = (box(B2,B2));
BB=ndSparse(BB);
F = (rho(BB,F,2));
clear BB;
BB = (box(BasisTimedomain,BasisTimedomain));
F = (rho_p3(BB,F,3));

R = rho(B1,seq,1);
R = rho(B2,R,2);
R = rho(BasisTimedomain,R,3);
R = reshape(R,[n1*n2*n3,1]);

F = reshape(F,[n1,n1,n2,n2,n3,n3]);
F = permute(F,[1,3,5,2,4,6]); 
F = reshape(F,[n1*n2*n3, n1*n2*n3]);

% compute bspline coefficients and smoothed data
R = double(R);
A = F \ R;
Ar = reshape(A,[n1,n2,n3]);
Ar = rho_faster(B1',Ar,1);
Ar = rho_faster(B2',Ar,2);
Rdrift = rho_p3(BasisTimedomain',Ar,3); % slowest part

RSS = sum(sum(sum(    (seq - Rdrift).^2    ))) / (n*dt);
fprintf('dbg: rss %.6f\n',RSS);

RdMin = reshape(Rdrift,n,dt);

end



function [act, weights, t_S] = computeShutterPatternV3(seq,T)
ms = mean(seq);
dMS = diff(ms);
dT = diff(T);
t_S = find(dT > 510);
fprintf('Shutter pattern > 510!\n');

%t_S = locateNUCpointsNew3(dT,dMS,mean(seq))'; %auskommentiert 27.11.2014
%t_S = locateNUCpointsNew(dT)';% auskommentiert 12.12.
n = length(t_S);

if(n == 1)
    weights(1:length(T)) = 1;
    act(1:n,1:length(T)) = -1;
    act(1,t_S(1):end) = 1;    
    return 
end

weights(1:length(T)) = 1;
act(1:n,1:length(T)) = 0;

act(1,1:t_S(1)-1) = -1;
act(1,t_S(1):t_S(2)-1) = 1;

dWeights = 5;
weights(t_S(1) - dWeights : t_S(1)+dWeights) = 0;
for i=3:n
    weights(t_S(i-1) - dWeights : t_S(i-1)+dWeights) = 0;
    act(i-1,t_S(i-2):t_S(i-1)) = -1;
    act(i-1,t_S(i-1):t_S(i)-1) = 1;   
end
weights(t_S(n) - dWeights : t_S(n)+dWeights) = 0;
act(n,t_S(n-1):t_S(n)-1) = -1;
act(n,t_S(n):end) = 1;

end


function C = rho(A,B,p) % from 2003
    sa = size(A);
    sb = size(B);
    n = length(sb);
    ip = [(p+1):n,1:(p-1)];
    B = permute(B,[p,ip]);
    B = reshape(B,[sb(p),prod(sb(ip))]);
    C = A'*B;
    C = reshape(C,[sa(2),sb(ip)]);
    clear At;
    clear B;
    C = ipermute(C,[p,ip]);
end


function C = rho_p3(A,B,p) % from 2003
    sa = size(A);
    sb = size(B);
    n = length(sb);
    ip = [(p+1):n,1:(p-1)];
    %B = permute(B,[p,ip]);
    B = reshape(B,[prod(sb(ip)),sb(p)]);
    C = B*A;
    C = reshape(C,[sb(ip),sa(2)]);
    clear At;
    clear B;
    %C = ipermute(C,[p,ip]);
end

function C2 = rho_faster(A,B,p)
    sa = size(A);
    sb = size(B);
    n = length(sb);
    ip = [(p+1):n,1:(p-1)];
    B = permute(B,[ip,p]);
    B = reshape(B,[prod(sb(ip)),sb(p)]);
    
    C2 = B*A;
    C2 = reshape(C2,[sb(ip), sa(2)]);
    
    C2 = ipermute(C2,[ip,p]);
%     At = A';
%     clear A;
%     C = At*B;
%     C = reshape(C,[sa(2),sb(ip)]);
%     clear At;
%     clear B;
%     C = ipermute(C,[p,ip]);
end

% % % 
function [P] = compute3DDifferencePenalty(n1,n2,n3,d1,d2,d3,l1,l2,l3)
E1 = (eye(n1));
E2 = (eye(n2));
E3 = (eye(n3));

D1 = (diff(E1,d1));
D2 = (diff(E2,d2));
D3 = (diff(E3,d3));

P1 = (kron((kron(l1 .* (D1' * D1),E2)),E3));
P2 = (kron((kron(E1,l2 .* (D2'*D2))),E3));
P3 = (kron((kron(E1,E2)),l3 .* (D3' * D3)));
P = P1 + P2 + P3;
end

function [P] = compute3DDifferencePenaltyBroken(n1,n2,n3,n3BasisElements,d1,d2,d3,l1,l2,l3)
nOtherElements = n3 - n3BasisElements;

E1 = (eye(n1));
E2 = (eye(n2));
E3 = 0 .* (eye(n3));
%E3(n3BasisElements+1:end) = 0;
D1 = (diff(E1,d1));
D2 = (diff(E2,d2));
E3BasisOnly = eye(n3BasisElements);
D3 = blkdiag(  diff(E3BasisOnly,d3)'*diff(E3BasisOnly,d3), zeros(nOtherElements,nOtherElements) );
P1 = (kron((kron(l1 .* (D1' * D1),E2)),E3));
P2 = (kron((kron(E1,l2 .* (D2'*D2))),E3));
P3 = (kron((kron(E1,E2)),l3 .* D3));
P = P1 + P2 + P3;
end

function [P] = compute3DDifferencePenaltyBroken2(n1,n2,n3,n3BasisElements,d1,d2,d3,l1,l2,l3)
nOtherElements = n3 - n3BasisElements;

E1 = (eye(n1));
E2 = (eye(n2));
E3 = (eye(n3));
%E3(n3BasisElements+1:end,n3BasisElements+1:end) = 0;
D1 = (diff(E1,d1));
D2 = (diff(E2,d2));
E3BasisOnly = eye(n3BasisElements);
D3 = blkdiag(  diff(E3BasisOnly,d3)'*diff(E3BasisOnly,d3), zeros(nOtherElements,nOtherElements) );
P1 = (kron((kron(l1 .* (D1' * D1),E2)),E3));
P2 = (kron((kron(E1,l2 .* (D2'*D2))),E3));
P3 = (kron((kron(E1,E2)),l3 .* D3));
P = P1 + P2 + P3;
end

function b = box(B1,B2)


[n11,n12] = size(B1);
[n21,n22] = size(B2);

b = kron(ones(n12,1)',B1) .* (kron(B2,ones(n22,1)'));

end
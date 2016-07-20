function [ RdMin, dfs, AICs ] = PSplineSmoothing3dImgsequence( seq, seq_timing )
[n,dt] = size(seq);
noElem = n * dt;

[shutterPtnrV3] = computeShutterPatternV3(seq,seq_timing);
%[shutterPtnrV2] = computeShutterPatternV2(seq_timing);

shutterPtnrV3 = shutterPtnrV2;

%seq = double(seq);
seq = reshape(seq,[480,640,dt]);

[B1,Dx] = computePenBSplineBasis(480,3,2,10); % 25 25 20
[B2,Dy] = computePenBSplineBasis(640,3,2,10);
[Bdrift,Draw] = computePenBSplineBasis(dt,3,2,100);
BasisTimedomain = [Bdrift shutterPtnrV3'];
%BasisTimedomain = Bdrift;
[~,n1] = size(B1);
[~,n2] = size(B2);
[~,n3] = size(BasisTimedomain);
[~,n3BSpline] = size(Bdrift);

W = (ones(480,640,dt));
BB = (box(B1,B1));
F = (rho(BB,W,1));
clear BB;
clear W;

BB = (box(B2,B2));
F = (rho(BB,F,2));
clear BB;
BB = (box(BasisTimedomain,BasisTimedomain));
F = (rho_p3(BB,F,3));

%P = compute3DDifferencePenalty(n1,n2,n3,2,2,2,1000,1000,0.1);
%P1 = compute3DDifferencePenaltyBroken(n1,n2,n3,n3BSpline,2,2,2,1000,1000,0.1);
%P2 = compute3DDifferencePenaltyBroken2(n1,n2,n3,n3BSpline,2,2,2,1000,1000,0.1);
%P1 = P2;
R = rho(B1,seq,1);
R = rho(B2,R,2);
R = rho(BasisTimedomain,R,3);
R = reshape(R,[n1*n2*n3,1]);

F = reshape(F,[n1,n1,n2,n2,n3,n3]);
F = permute(F,[1,3,5,2,4,6]);
F = reshape(F,[n1*n2*n3, n1*n2*n3]);
% % % % G = inv(F + P1);
% % % % clear P;
% % % % A = G * R;
% % % % Ar = reshape(A,[n1,n2,n3]);
% % % % Ar = rho_faster(B1',Ar,1);
% % % % Ar = rho_faster(B2',Ar,2);
% % % % Rdrift = rho_p3(BasisTimedomain',Ar,3); % slowest part
% % % % %Rdrift = reshape(Rdrift,480*640,dt);

%lambdas = [1 10 100 1000];
lambdas = linspace(0.01,20,10);
dfs(1:length(lambdas)) = 0;
AICs(1:length(lambdas)) = 0;
RdMin = 0;
AICmin = 10000000;
for i=1:length(lambdas)
    tic;
    % penalty matrix
    li = lambdas(i);
    %lis = 0.1 *  1/li;
    lis = 1/li;
    %P1 = compute3DDifferencePenaltyBroken(n1,n2,n3,n3BSpline,2,2,2,li,li,lis);
    %P1 = compute3DDifferencePenaltyBroken2(n1,n2,n3,n3BSpline,2,2,2,li,li,lis);
    P1 = compute3DDifferencePenalty(n1,n2,n3,3,3,3,li,li,lis);
    
    % compute bspline coefficients and smoothed data
    A = (F+P1) \ R;
    Ar = reshape(A,[n1,n2,n3]);
    Ar = rho_faster(B1',Ar,1);
    Ar = rho_faster(B2',Ar,2);
    Rdrift = rho_p3(BasisTimedomain',Ar,3); % slowest part
    
    % compute trace of hat matrix for DoF estimation
    G = inv(F + P1);
    G = reshape(G,[n1,n2,n3,n1,n2,n3]);
    G = permute(G,[1,4,2,5,3,6]);
    G = reshape(G,[n1*n1,n2*n2,n3*n3]);
    H = rho(box(B1,B1)',G,1);
    H = rho(box(B2,B2)',H,2);
    H = rho(box(BasisTimedomain,BasisTimedomain)',H,3);
    df = sum(H(:));
    dfs(i) = df; 
    RSS = sum(sum(sum(    (seq - Rdrift).^2    )));
    AIC = log(RSS) + (2 * (df+1)) / (noElem-df-2) ;
    toc;
    if(AIC < AICmin)
       AICmin = AIC; 
       RdMin = Rdrift;
       fprintf(' ** ');
    end
    
    AICs(i) = AIC;
    fprintf('l1 %.3f l2 %.3f ls %.3f df %.3f AIC %.6f\n',li,li,lis,df, AIC);
end

RdMin = reshape(RdMin,n,dt);

end



function [act] = computeShutterPatternV2(T)
dT = diff(T);
t_S = find(dT > 200);
%t_S = locateNUCpointsNew(dT)';
if(t_S(1) == 1)
    t_S = t_S(2:end);
end
act(1:length(t_S)+1,1:length(T)) = 0;
act(1,1:t_S(1)-1) = 1;

for i=2:length(t_S)
    act(i,t_S(i-1):t_S(i)-1) = 1;
end

act(length(t_S)+1,t_S(end):end) = 1;

end



function [act, weights] = computeShutterPatternV3(seq,T)
ms = mean(seq);
dMS = diff(ms);
dT = diff(T);
t_S = find(dT > 200);

%t_S = locateNUCpointsNew3(dT,dMS,mean(seq))'; %auskommentiert 27.11.2014
%t_S = locateNUCpointsNew(dT)';% auskommentiert 12.12.
n = length(t_S);
figure();plot(1:length(ms),ms,t_S,ms(t_S),'r*',t_S,ms(t_S),'r-');drawnow;
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
function [ F,B1,B2,Bdrift ] = prepare3DPSplineBasis( dx,dy,dt )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


[Bdrift,Draw] = computePenBSplineBasis(dt,4,2);
[B1,Dx] = computePenBSplineBasis(dx,2,2,20);
[B2,Dy] = computePenBSplineBasis(dy,2,2,20);

[~,n1] = size(B1);
[~,n2] = size(B2);
[~,n3] = size(Bdrift);

W = (ones(480,640,dt));
BB = (box(B1,B1));
F = (rho(BB,W,1));
clear BB;
clear W;

BB = (box(B2,B2));
F = (rho(BB,F,2));
clear BB;
BB = (box(Bdrift,Bdrift));
F = (rho(BB,F,3));


% [Bdrift,Draw] = computePenBSplineBasis(dt,4,2);
% [B1,Dx] = computePenBSplineBasis(dx,2,2,20);
% [B2,Dy] = computePenBSplineBasis(dy,2,2,20);
% 
% [~,n1] = size(B1);
% [~,n2] = size(B2);
% [~,n3] = size(Bdrift);
% 
% W = single(ones(480,640,dt));
% BB = single(box(B1,B1));
% F = single(rho(BB,W,1));
% clear BB;
% clear W;
% 
% BB = single(box(B2,B2));
% F = single(rho(BB,F,2));
% clear BB;
% BB = single(box(Bdrift,Bdrift));
% F = single(rho(BB,F,3));

end


function C = rho(A,B,p)
    sa = size(A);
    sb = size(B);
    n = length(sb);
    ip = [(p+1):n,1:(p-1)];
    B = permute(B,[p,ip]);
    B = reshape(B,[sb(p),prod(sb(ip))]);
    At = A';
    clear A;
    C = At*B;
    C = reshape(C,[sa(2),sb(ip)]);
    clear At;
    clear B;
    C = ipermute(C,[p,ip]);
end

function b = box(B1,B2)


[n11,n12] = size(B1);
[n21,n22] = size(B2);

b = kron(ones(n12,1)',B1) .* (kron(B2,ones(n22,1)'));

end
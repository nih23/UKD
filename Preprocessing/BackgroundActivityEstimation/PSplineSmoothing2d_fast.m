function [ imgs, imgsNoBg ] = PSplineSmoothing2d_faster(  img, lambda )

[n1,n2] = size(img);
noPx = n1 * n2;


l1 = 10000;

if(exist('lambda','var'))
    l1 = lambda;
end

l2 = l1;
fprintf('* 2D penalized B-Spline image smoothing w/ l1=%.2f, l2=%.2f\n',l1,l2);
fprintf('  2nd order B-Spline and 2nd order penalty. Eilers Approach.\n');
%[B1,Dx] = computePenBSplineBasis(n1,4,2);
%[B2,Dy] = computePenBSplineBasis(n2,4,2);
kn1 = min([n1/4,200]);
kn2 = min([n2/4,200]);
[B1,Dx] = computePenBSplineBasis(n1,2,2,kn1);
[B2,Dy] = computePenBSplineBasis(n2,2,2,kn2);


% tiling base
ycos = diag(cos((2*pi*1/20) .* [1:n2]));
ysin = diag(sin((2*pi*1/20) .* [1:n2])); 
Bcos = ycos * B2;
Bsin = ysin * B2;
[~,m1Small] = size(Bcos);

B2 = [Bcos Bsin B2];
[~,m1] = size(B1);
[~,m2] = size(B2);
lModulation = l1;
%P = lModulation .* kron(eye(m1Small),Dx) + lModulation .* kron(eye(m1Small),Dx) + l1 .* kron(eye(m1Small),Dx) + l2 .* kron(Dy,eye(m2));
%P = lModulation .* kron(eye(m1),Dx) + l2 .* kron(Dy,eye(m2));
P = computeDifferencePenalty(m1,m2,2*m1Small,3,3,l1,l2,100*l2);
%% FAST IMPLEMENTATION
W = ones(n1,n2);

F0 = boxNew(B1,B1)' * W * boxNew(B2,B2);
F1 = reshape(F0,[m1,m1,m2,m2]);
F2 = permute(F1,[1,3,2,4]);
F3 = reshape(F2,[m1*m2,m1*m2]);
clear F0 F1 F2 W;
iT = inv(F3+P);

   
    R = B1'*img*B2;
    r = R(:);
    a = iT * r;
    A = reshape(a,[m1,m2]);
    
    imgs = reshape(B1*A*B2',[n1 n2]);
    
    A(:,2*m1Small+1:end) = 0;
    imgsNoBg = reshape(B1*A*B2',[n1 n2]);


end

function b = boxNew(B1,B2)


[n11,n12] = size(B1);
[n21,n22] = size(B2);

b = kron(ones(n12,1)',B1) .* (kron(B2,ones(n22,1)'));

end

function B = box(B1,B2)

[J,L] = size(B1);
[I,K] = size(B2);

B = kron(B1,ones(1,L)) .* kron(ones(1,K),B2);

end

function [P] = computeDifferencePenalty(n1,n2,n2last, d1,d2,l1,l2,l3)
E1 = (eye(n1));
E2 = (eye(n2));
D1 = (diff(E1,d1));
D2 = (diff(E2,d2));

P1 = kron(l1 .* (D1' * D1),E2);
D2tD2 = D2' * D2;
D2tD2(1:n2last,1:n2last) = l3 .* D2tD2(1:n2last,1:n2last);
D2tD2(n2last+1:end,n2last+1:end) = l2 .* D2tD2(n2last+1:end,n2last+1:end);
P2 = kron(E1,D2tD2);

P = P1 + P2;
end
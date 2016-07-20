function [ seqn ] = PSplineSmoothing2dImgsequenceFaster(  seq, lambda )

[noPx,noFrames] = size(seq);

n1 = 480;
n2 = 640;

l1 = 10000;

if(exist('lambda','var'))
    l1 = lambda;
end

l2 = l1;
fprintf('* 2D penalized B-Spline image smoothing w/ l1=%.2f, l2=%.2f\n',l1,l2);
fprintf('  2nd order B-Spline and 2nd order penalty. Eilers Approach.\n');
%[B1,Dx] = computePenBSplineBasis(n1,4,2);
%[B2,Dy] = computePenBSplineBasis(n2,4,2);

[B1,Dx] = computePenBSplineBasis(n1,2,2);
[B2,Dy] = computePenBSplineBasis(n2,2,2);
[~,m1] = size(B1);
[~,m2] = size(B2);

%P = l1 .* kron(eye(L),Dx'*Dx) + l2 .* kron(Dy'*Dy,eye(K));
P = l1 .* kron(eye(m1),Dx) + l2 .* kron(Dy,eye(m2));
%seq = double(seq);
%seqnew = seq;
% %% FAST IMPLEMENTATION
W = ones(n1,n2);

 F0 = boxNew(B1,B1)' * W * boxNew(B2,B2);
 F1 = reshape(F0,[m1,m1,m2,m2]);
 F2 = permute(F1,[1,3,2,4]);
 F3 = reshape(F2,[m1*m2,m1*m2]);
 clear F0 F1 F2 W; 
 iT = inv(F3+P);
 YY = reshape(seq,480,640,noFrames);
 seqn = seq;
 parfor i=1:noFrames
     Y = squeeze(YY(:,:,i));
     R = B1'*Y*B2;
     r = R(:);
     a = iT * r;
     A = reshape(a,[m1,m2]);
     img = reshape(B1*A*B2',[n1*n2 1]);
     seqn(:,i) = img;
 end


end

function b = boxNew(B1,B2)


[n11,n12] = size(B1);
[n21,n22] = size(B2);

b = kron(ones(n12,1)',B1) .* (kron(B2,ones(n22,1)'));

end
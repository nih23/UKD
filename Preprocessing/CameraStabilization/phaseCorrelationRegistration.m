function [ seqR ] = phaseCorrelationRegistration( seq )
% Summary of this function goes here
%   Detailed explanation goes here

[n,m] = size(seq);
seqR(1:n,1:m) = seq;

%fixed = reshape(seq(:,1),480,640);
imgA = mat2gray(reshape(seq(:,1),480,640));
Rfixed = imref2d(size(imgA));

for i=2:m
   fprintf('%d/%d\n',i,m);
   %moving = reshape(seq(:,i),480,640); 
   imgB = mat2gray(reshape(seq(:,i),480,640)); 
   
   
   tformEstimate = imregcorr(imgB,imgA,'similarity');
   %sum(sum(tformEstimate.T))
   movingReg = imwarp(reshape(seq(:,i),480,640),tformEstimate,'OutputView',Rfixed);
   seqR(:,i) = movingReg(:);
end

end


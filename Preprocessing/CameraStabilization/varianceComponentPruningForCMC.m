function [ seqFVCP ] = varianceComponentPruningForCMC( seq, framerate, useOneSVM,dx,dy )
%VARIANCECOMPONENTPRUNINGFORCMC Compensation of quasi-periodic
%high-frequency camera vibrations using Wavelet-VCP approach.
%PARAMETERS:
%               seq: n x m sequence of n length m timeseries (m must be dyadic)
%               framerate: recording framerate (default: 50) in fps
%               minimumVibrationFrequency: expected lower bound of
%               vibration frequency (default: 2) in Hz
[~,m] = size(seq);
assert(m == 2^nextpow2(m),'Each timeseries has to have dyadic length!');
%assert(n == 307200,'Require 640x480 Px. recording!');

img = reshape(seq(:,1),dx,dy);
[Fx,Fy] = gradient(img);
F = abs(Fx) + abs(Fy);
[~,si] = sort(F(:),'descend');
noEdges = 1000;
[strongestGradients, weakestGradients] = evalGradientCandidates(si,dx,dy,noEdges);
%strongestGradients = si(1:noEdges);
%weakestGradients = si(end-noEdges:end);
dyadM = nextpow2(m);
%fprintf('Camera motion correction using WaveletVCP\n');
qmf = MakeONFilter('Symmlet',4);
%fprintf('Forward wavelet transform\n');
wc = FWT_PO_SEQUENCE(seq,1,qmf);
clear seq;
%minimumVibrationFrequency = 1;
%for j = waveletScaleForTargetFrequency(framerate,dyadM,1) : dyadM-1   

for j = 3 : dyadM-1   
    %fprintf('** dyad %d\n',j);
    wcJ = wc(:,dyad(j)); 
    wcJ = vcpCMC(wcJ,strongestGradients, weakestGradients, useOneSVM);  
    wc(:,dyad(j)) = wcJ;
    clear wcJ;
end
% wc(:,idxxxx) = vcpCMC(wc(:,idxxxx),3,strongestGradients, weakestGradients);
%fprintf('Inverse wavelet transform\n');
seqFVCP = IWT_PO_SEQUENCE(wc,1,qmf);

end

function [ xct ] = vcpCMC( seqSS, strongestGradients, weakestGradients, useOneSVM )
%VARIANCECOMPONENTPRUNING Prunes maximum influencial variance components of
%given datamatrix.
%% data preparation
[~,m] = size(seqSS);
mu = sum(seqSS,2)/m;
% mean centering
xc = bsxfun(@minus,seqSS,mu);
clear seqSS;
% if(exist('normVariance','var'))
% xcvar = var(xc,1,2)*ones(1,m);
% xc = xc ./ xcvar;
% end

% norm variance?
%% compute eigenvectors of mean centered data and prune specified leading eigenvectors
% assumption: eigenvectors with maximum eigenvalue rests as last element of
% V
[V, ~] = eig(xc'*xc);
%figure();imagesc(reshape(abs(xc(:,end-3)),480,640))
xct = (xc*V);
n = length(V);
xcStr = sum(abs(xct(strongestGradients,:)));% ./ n;
xcWeak = sum(abs(xct(weakestGradients,:)));% ./ n;
xcStrWeakRatio = xcStr./xcWeak;
%xcStrWeakRatio = (xcStrWeakRatio / max(xcStrWeakRatio))';



%[B,S] = lasso(A,xcStrWeakRatio','CV',m);

% % % figure();plot((sort(xcStrWeakRatio)));
if(useOneSVM == 0)
    [class,~] = dbscan(xcStrWeakRatio,30,[]);
    idxSelPx = find(class == -1);
elseif(useOneSVM == 1)
    idxSelPx = oneSVM(xcStrWeakRatio)';
    %idxSelPx = oneSVM([xcStr; xcWeak])';
elseif(useOneSVM == 2)
    A = [1:m]';    
    w = warning ('off','all');
    [b,stats] = robustfit(A,xcStrWeakRatio);
    theta = tcdf(abs(stats.rstud),stats.dfe);
    alpha = 0.05;
    i1 = find(theta > 1-alpha);
    %i2 =  find(theta < alpha);
    i2 = [];
    idxSelPx = [i1; i2];
%     
%     sigm3Upper = b'*[ones(m,1) A]' + 2*stats.robust_s;
%     sigm3Lower = b'*[ones(m,1) A]' - 2*stats.robust_s;
%     i1 = find(xcStrWeakRatio > sigm3Upper)';
%     i2 = find(xcStrWeakRatio < sigm3Lower)';
%    idxSelPx = [i1; i2];
    %idxSelPx = [i1];
    w = warning ('on','all');
    %figure();plot(1:m,xcStrWeakRatio,1:m,b'*[ones(m,1) A]',1:m,b'*[ones(m,1) A]' + 2*stats.robust_s, 1:m,b'*[ones(m,1) A]' - 2*stats.robust_s);   
else
    idxSelPx = [m m-1 m-2 m-3 m-4];
end

%idxSelPx = [idxSelPx m m-1 m-2];

clear xc;
if(~isempty(idxSelPx))   
    figure();plot(1:length(xcStr),xcStrWeakRatio,idxSelPx,xcStrWeakRatio(idxSelPx),'ro');title(sprintf('%d',length(xcStr)));drawnow;  
    xct(:,idxSelPx) = 0;
else
   fprintf('Couldnt estimate motion artifacts for current scale! Aborting.\n');
   %figure();plot(1:length(xcStr),xcStrWeakRatio);title(sprintf('%d',n));drawnow;
end

MPinv = (V'*V)\V'; % moore penrose pseudo inverse
clear V;
xct = xct * MPinv;
xct = bsxfun(@plus,xct,mu);

end

function [idxOutlier] = oneSVM(X)
X = X';
y = ones(size(X,1),1);
rng(1);
SVMModel = fitcsvm(X,y,'KernelScale','auto','Standardize',false,'OutlierFraction',0.1);
[~,score] = predict(SVMModel,X);
idxOutlier = find(score < 0);

%idxOutlier = idxOutlier(find(X(idxOutlier > 0.1)));
    

end

function [loss] = huberLoss(data,thresh)

    a = data-thresh;
    idx1 = find(abs(a) < thresh);
    idx2 = find(a >= thresh);
    loss = sum(1/2*a(idx1).^2) + sum(thresh * (abs(a(idx2))-thresh/2));

end
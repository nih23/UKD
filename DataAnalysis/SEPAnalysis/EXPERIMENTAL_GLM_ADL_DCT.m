function [ t_OLS, t_HC2,t_HC4, t_NW ] = GLMExperimentalADLDCT( seq, seq_timing, patientName )
%GLM Fits simple Y = Gb + e model to data, with G := [g_1^T 1^T] and g_1
%being the stimulation sequence
% seq: sequence to fit glm to (n x m matrix with n pixels)
% framerate: frames per second
% stim_response_offset: offset between stimulation time and expected
% response time in seconds (default: 7s)
% contact: nico.hoffmann@tu-dresden.de
%%%fprintf('*  GLM-based SEP detection\n');
% % %% prepare model

t_start = seq_timing(1);
[n,m] = size(seq);

seq = double(seq);
fprintf('*** SEP ANALYSIS ***\n \n');

seq = seq';

dx = 480;
dy = 640;

stim_width = 30;
%fprintf('stim delay %d s stim width %d s\n',stim_response_offset,stim_width);
t_stim = 30; %seconds

qmf = MakeONFilter('Symmlet',4);
L=1;
XI=eye(m);
W=zeros(m); % DWT BASIS
for i=1:m
    W(:,i)=FWT_PO(XI(:,i),L,qmf);
end

GwithoutStim = [W(dyad(1),:)' W(dyad(2),:)' W(dyad(3),:)'];



%% VCM modulation model
[fPulse, fResp] = estimatePulseFrequency(seq_timing,seq);
seq = double(seq);
fPulseMs = 60 / fPulse * 1000;
fRespMs = 60 / fResp * 1000;
[Braw300,Draw300] = computePenBSplineBasis(m,4,3,100); 
[Braw100,Draw100] = computePenBSplineBasis(m,4,3,100); 
%ycos = diag(cos((2*pi*1/1000) .* seq_timing)); % PULS
%ysin = diag(sin((2*pi*1/1000) .* seq_timing)); % PULS

ycos = diag(cos((2*pi*1/fPulseMs) .* seq_timing)); 
ysin = diag(sin((2*pi*1/fPulseMs) .* seq_timing)); 

ycosAtmung = diag(cos((2*pi*1/fRespMs) .* seq_timing)); % ATMUNG
ysinAtmung = diag(sin((2*pi*1/fRespMs) .* seq_timing)); % ATMUNG


BcosPuls = ycos * Braw300;
BsinPuls = ysin * Braw300;

BcosAtmung = ycosAtmung * Braw100;
BsinAtmung = ysinAtmung * Braw100;

%% F-TEST / penalized B-spline OLS

%  fprintf('boxcar activity\n');
%  stimPtnr = computeBoxCarActivityPatternV1(0, t_stim, stim_width, t_start, seq_timing,30);
%  stimPtnr(find(stimPtnr == 0)) = -1;
%  stimPtnr = stimPtnr';
fprintf('Gaussian activity\n');
stimPtnr = computeGaussianActivityPatternV1(0, t_stim, stim_width, t_start, seq_timing);
%stimPtnr = stimPtnr ./ max(stimPtnr);



%fprintf('sinus activity\n');
%F = fft(eye(m));
%Fs = 10;
%NFFT = 2^nextpow2(5329);
%[~,idx] = min(abs(Fs/2*linspace(0,1,NFFT/2+1) - 1/60))
%fAct = F(idx,:);
%dS = [diff(stimPtnr) 0];
D = dctmtx(m);
%G = [stimPtnr' GwithoutStim ones(m,1)]; % dS' 
%G = [stimPtnr' D(1:5,:)']; % oder 1:10 !
G = [stimPtnr' D(:,2:50)]; % oder 1:10 !
beta = G \ seq;
covG = inv(G'*G);
covStim = covG(1,1);
[~,df] = size(G);
RSS = computeResSSQ_BIGMEM(seq,G,beta);
sigma_e = RSS ./ (m - df - 1);
t = beta(1,:) ./ sqrt(covStim * sigma_e);


figure();imagesc(reshape(t,480,640));title('t-OLS');drawnow;



return






%% OLD OLD OLD
%b1 = (G'*W*G + D)\G'*W; % ridge regression
b1 = (G'*G + D)\G'; % ridge regression
beta = b1*seq;
Slambda = G*b1;
df = trace(Slambda);
RSS = computeResSSQ_BIGMEM(seq,G,beta);
GCVlambda = RSS ./ ((1-df/m)^2);
AIClambda = log(RSS) + 2 * df / m;
df_res = m - 2*df - trace(Slambda' * Slambda); % Ruppert, Wand, Carroll 2003
fprintf('>> df = %f lambda = %f GCV(lambda) = %f AIC(lambda) = %f\n',df,lambda,mean(GCVlambda),mean(AIClambda));


pBonf = 0.05/2;
Fbonf = finv(1-pBonf,1,m-df-1);
tbonf = tinv(1-pBonf,m-df-1);
fprintf('>> t = %.3f Fbonf = %.3f\n',tbonf, Fbonf);
pBonf = 0.05/(2*307200);
Fbonf = finv(1-pBonf,1,m-df-1);
tbonf = tinv(1-pBonf,m-df-1);
fprintf('>> tBonf = %.3f Fbonf = %.3f\n',tbonf, Fbonf);

%ssq1 = RSS ./ (m-df-1);
ssq1 = RSS ./ df_res; % Ruppert, Wand, Carroll 2003
%ssq1 = computeSigmaIID_BIGMEM(seq,G,beta);
[~,degreeOfFreedom] = size(G);
c(1:degreeOfFreedom) = 0;
c(1) = 1;
t_OLS_pen = (c * beta) ./ sqrt((c/(G'*G))*c'*ssq1);
figure();imagesc(reshape(t_OLS_pen,480,640));title('OLS');drawnow;

tic;t_ARp=FGLS_CO_ARp_PSpline(seq,G,D,lambda);toc;
figure();imagesc(reshape(t_ARp,480,640));title('AR');drawnow;
rr;
regre = G*beta;

 

%eGlobal = (seq - regre)';




G = GwithoutStim;
[jj,kk] = size(G);
D = blkdiag(zeros(kk,kk),Draw);
G = [G Bdrift];
beta = inv(G'*G + lambda*D)*G'*seq;
ssq2 = computeSigmaIID_BIGMEM(seq,G,beta);
F = ((ssq2 - ssq1) ./ (degreeOfFreedom - rank(G'*G))) ./ (ssq1 ./ (m - degreeOfFreedom)) ; 
toc;


%% t-Test / penalized OLS

% tic;
% G = [stimPtnr GwithoutStim];
% G = double(G);
% [jj,kk] = size(G);
% D = blkdiag(zeros(kk,kk),D);
% G = [G C];
% beta = inv(G'*G + 10000*D)*G'*seq;
% %beta = computeBetaFast(seq,G);
% ssq1 = computeSigmaIID_BIGMEM(seq,G,beta);
% [nAll,mAll] = size(G);
% [n0,m0] = size(G);
% clear c;
% c(1:mAll) = 0;
% c(1) = 1;
% t_OLS_pen = (c * beta) ./ sqrt((c/(G'*G))*c'*ssq1);
% toc;

%% t-Test / OLS
tic;
G = [stimPtnr GwithoutStim ones(m,1)];
G = double(G);
beta = computeBetaFast(seq,G);
ssq1 = computeSigmaIID_BIGMEM(seq,G,beta);
[nAll,mAll] = size(G);
[n0,m0] = size(G);
clear c;
c(1:mAll) = 0;
c(1) = 1;
t_OLS = (c * beta) ./ sqrt((c/(G'*G))*c'*ssq1);
toc;

t_CO(1:n) = 0;
% tic;
% [t_CO] = FGLS_ARp_WORSLEY(seq,G,beta);
% toc;

%% t-Test / TIME-VARIATION
% tic;
% [ ee, tt ] = GLMSEPevalTimeshifts( seq, GwithoutStim, t_stim,stim_width,t_start,seq_timing, c );
% [~,timeShiftIdx] = min(ee);
% t_Timeshift = selectVaryingRowsFromMatrix(tt,timeShiftIdx);
t_Timeshift(1:307200) = NaN;
% toc;

%% t-Test / HC
tic;
[t_HC1,t_HC2,t_HC3,t_HC4,se,se_HC2,se_HC3,se_HC4] = computeTWhiteSparse(seq,G,beta,c);
toc;
%clear c;

% tic;
% [t_CO,sigma,rho,PWbeta] = FGLS_CO_BIGMEM(seq,G,beta);
% toc;

t_FGLS(1:n) = 0;
t_NW(1:n) = 0;
t_FGLS(1:n) = 0;

%tic;
%[beta,sigma,t_FGLS] = FGLS(seq,G,beta,c);
%toc;


%Ij = diag(ones(1,m));
%R = Ij - G*pinv(G);
%R0 = Ij - G0*pinv(G0);
%M = R0 - R;
%F = ((seq' * M * seq) ./ (seq' * R * seq)) * ((J-mAll)/m0);
%R1 = computeR(seq,ssq1);
% F test under IID assumptions -> IST SSQ korrekt?
%F_het = ((se_red - se) ./ (degreeOfFreedom - rank(G'*G))) ./ (se ./ (m - degreeOfFreedom)) ; % Wald test under Heteroscedasticity assumptions


%figure();subplot(1,2,1);imagesc(reshape(F,dx,dy));title('F raw');
%subplot(1,2,2);imagesc(reshape(F ./ std(beta_stim(1:mStim,:)),dx,dy));title('F scaled by beta_std');
%fprintf('');
%fc = fcdf(abs(F),1,m-degreeOfFreedom-1);
%% t-test w/ bonferoni correction and euler characteristic evaluation
%%%fprintf('   pBonf = 0.05 / n\n');
% n_labels = length(find(labels == 1));
% pBonf = 0.05 / n_labels;
% Fbonf = finv(1-pBonf,1,m-degreeOfFreedom-1);
% tbonf = tinv(1-pBonf,m-degreeOfFreedom-1);
% 
% activity_f(1:n) = single(0);
% activity_f(find(F > Fbonf)) = 1;
% 
% activity_t(1:n) = single(0);
% activity_t(find(t > tbonf)) = 1;
% 
% 
% %%%fprintf('   Euler characteristics\n');
% sigma = 10;
% H = fspecial('gaussian', [10 10], sigma);
% Ffiltered = reshape(F,dx,dy);
% Ffiltered = imfilter(Ffiltered,H,'replicate');
% Ffiltered = Ffiltered(:);
% tfiltered = reshape(t,dx,dy);
% tfiltered = imfilter(tfiltered,H,'replicate');
% tfiltered = tfiltered(:);
% activity_f(find(Ffiltered > Fbonf)) = 1;
% activity_f_ec(1:n) = NaN;

%figure();imagesc(reshape(Ffiltered,480,640));title(sprintf('F-filtered @VCP%d',noExpectedDriftComponents));
%figure();imagesc(reshape(tfiltered,480,640));title('t-filtered');drawnow();

% FecL = floor(10*finv(0.95,1,m-degreeOfFreedom-1));
% 
% F(labels == 0) = 0;
% t(labels == 0) = 0;
% Ffiltered(labels == 0) = 0;
% tfiltered(labels == 0) = 0;
% 
% Fec = 0;
% ec_f(1:1000-FecL+1) = NaN;
% for i=FecL:1000
%     af(1:n) = single(0);
%     thres = i / 10;
%     af(find(Ffiltered > thres)) = 1;
%     af = reshape(af,dx,dy);
%     bin = imopen(af,ones(3,3));
%     bwe = imEuler2d(bin);
%     if(bwe > max(ec_f))
%         activity_f_ec = af(:);  
%         Fec = thres;
%     end
%     if(isempty(bwe))
%         bwe = 0;
%     end
%     ec_f(i-9) = bwe;  
% end
% 
% fprintf('   Fbonf = %.3f ; Fec = %.3f ; tbonf = %.3f\n',Fbonf,Fec,tbonf);
end

function [ ee, tt ] = GLMSEPevalTimeshifts( seq, GwithoutStimulation, t_stim,stim_width,t_start,seq_timing,c )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[~,degreeOfFreedom] = size(GwithoutStimulation);
degreeOfFreedom = degreeOfFreedom + 1;
[m,n] = size(seq);

ee(1:11,1:307200) = NaN;
tt(1:11,1:307200) = NaN;
for i=1:11
    tBeginActivity = i - 1;
    stimPtnr = computeBoxCarActivityPatternV1(tBeginActivity, t_stim, stim_width, t_start, seq_timing,30);
    G = [stimPtnr GwithoutStimulation ];
    G = double(G);
    beta = computeBetaFast(seq,G);
    regre = (seq-G*beta);
    ee(i,:) = sum( regre.^2);
    sigma = ee(i,:) ./ (m-degreeOfFreedom-1);
    tt(i,:) = (c * beta) ./ sqrt((c/(G'*G))*c'*sigma);
end

end


function [F] = computeTWhite(seq,G,beta,c)
fprintf('Heteroscedasticity Correction for t-statistics\n');
tic;
%[~,mAll] = size(G);
[m,n] = size(seq);
%beta_fgls = beta;
%bTg = beta'*G';
%GGinv = pinv(G'*G);
F(1:n) = 0;
XXTIX = c*( pinv(G'*G)*G');
XXTXI = (G*pinv(G'*G))*c';
%NN = 1:mAll;
for i=1:10000
    y_i = seq(:,i);
    beta_i = beta(:,i);
    u = (y_i - G*beta_i).^2;
    S = (diag(u));
    %S = sparse(NN,NN,u);
    varBeta = (sqrt((XXTIX * S * XXTXI)));
    F(i) = c*beta_i / varBeta;
    % White's heteroscedasticity-consistent estimator: http://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors
    %SwhiteINV = inv(diag(u.^2));
    %beta_fgls = pinv(G'*SwhiteINV*G)*G*SwhiteINV*y_i;
end
toc;
end

function [t,se] = computeTNeweyWest(seq,G,beta,c)
fprintf('Newey-West Autoregressive-Heteroscedasticity Variance Correction for t-statistics\n');

[~,n] = size(seq);
t(1:n) = 0;
se(1:n) = 0;

e = seq - G*beta;

tic;
parfor i=1:n
    %y_i = seq(:,i);
    beta_i = beta(:,i);
    %e = (y_i - G*beta_i);   
    e_i = e(:,i)
    nwse = NeweyWestFast(e_i,G);
    t(i) = (c*beta_i) / (c*nwse);
    se(i) = (c*nwse);
end
toc;




end



function [t_HC1,t_HC2,t_HC3,t_HC4,se,se_HC2,se_HC3,se_HC4] = computeTWhiteSparse(seq,G,beta,c)
fprintf('Heteroscedasticity Correction for t-statistics\n');
[m,n] = size(seq);
[~,k] = size(G);
t_HC1(1:n) = 0;
t_HC2(1:n) = 0;
t_HC3(1:n) = 0;
t_HC4(1:n) = 0;
XXTIX = (c*((G'*G)\(G')));
XXTXI = ((G/(G'*G))*c');
H = diag((G/(G'*G))*G');
D = (m*H)/k;
D(D>4) = 4;
se(1:307200) = 0;
se_HC2(1:307200) = 0;
se_HC3(1:307200) = 0;
se_HC4(1:307200) = 0;
parfor i=1:n
    y_i = seq(:,i);
    beta_i = beta(:,i);
    
    e = (y_i - G*beta_i);
    eSq = e.^2;
    cbi = c*beta_i;
    u = m/(m-k) * eSq; % HC1
    S = (sparse(1:length(u),1:length(u),u));
    varBeta = (sqrt((XXTIX * S * XXTXI)));
    se(i) = varBeta;
    t_HC1(i) = cbi / varBeta;
    
    u = eSq ./ (1-H); % HC2
    S = (sparse(1:length(u),1:length(u),u));
    varBeta = (sqrt((XXTIX * S * XXTXI)));
    t_HC2(i) = cbi / varBeta;
    se_HC2(i) = varBeta;
    
    u = (e ./ (1-H)).^2; % HC3
    S = (sparse(1:length(u),1:length(u),u));
    varBeta = (sqrt((XXTIX * S * XXTXI)));
    t_HC3(i) = cbi / varBeta;
    se_HC3(i) = varBeta;
    
    u = eSq ./ ((1-H).^D); % HC4
    S = (sparse(1:length(u),1:length(u),u));
    varBeta = (sqrt((XXTIX * S * XXTXI)));
    t_HC4(i) = cbi / varBeta;
    se_HC4(i) = varBeta;
end
end

function [F] = computeTWhiteGPU(seq,G,beta,c)
fprintf('White Heteroscedasticity Correction for t-statistics\n');
tic;
[m,n] = size(seq);
%beta_fgls = beta;
%bTg = beta'*G';
%GGinv = pinv(G'*G);
F(1:n) = 0;
XXTIX = gpuArray( pinv(G'*G)*G');
XXTXI = gpuArray(G*pinv(G'*G));
contr = gpuArray(c);
for i=1:1000
    y_i = gpuArray(seq(:,i));
    beta_i = gpuArray(beta(:,i));
    u = ((y_i - G*beta_i).^2);
    S = diag(u);
    varBeta = sqrt(contr*(XXTIX * S * XXTXI)*contr');
    F(i) = gather(contr*beta_i / varBeta);
    % White's heteroscedasticity-consistent estimator: http://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors
    %SwhiteINV = inv(diag(u.^2));
    %beta_fgls = pinv(G'*SwhiteINV*G)*G*SwhiteINV*y_i;
end
toc;
end



function [beta] = computeBeta(seq,G,useIRLS,mStim)
if(useIRLS == true)
   [beta,varData,v_ols] = mexRobustFit(single(seq),single(G));      
   return;
end 
[~,n] = size(seq);
[~,m] = size(G);
G = sparse(G);
beta(1:m,1:n) = 0;
if(exist('mStim','var'))
    % Tikhinov regularization
    [~, noBoringCoefficients] = size(G);
    D = blkdiag(diag(ones(mStim,1)), zeros(noBoringCoefficients-mStim,noBoringCoefficients-mStim));
    lambda = 0.1;
    %beta = ( pinv(G'*G ) * G') * seq;
    beta = ( pinv(G'*G + lambda*D) * G') * seq;
else
%    OLS
    %beta = G \ seq;
    G = sparse(G);
    [Q,R] = qr(G);
    Q = sparse(Q);
    R = sparse(R);
    %beta = ( (G'*G) \ G') * seq;
    %for i=1:n
    %   yi = seq(:,i);
    %   beta(:,i) = G \ yi;
    %end
    %beta = G \ seq;
    beta = (R \ Q') * seq;
end
   


end

function [beta] = computeBetaFast(seq,G)
%G = sparse(G);
beta = ( inv(G'*G) * G') * seq;
end

function [sSq] = computeResSSQ_BIGMEM(seq,G,beta)   
%e = computeSSQSinglePx(seq,G,beta);
[~,degreeOfFreedom] = size(G);
[m,n] = size(seq);
regre = G*beta;
eGlobal = seq - regre;
clear regre;
sSq = sum(eGlobal.^2,1);
end

function [sigma] = computeSigmaIID_BIGMEM(seq,G,beta)   
%e = computeSSQSinglePx(seq,G,beta);
[~,degreeOfFreedom] = size(G);
[m,n] = size(seq);
regre = G*beta;
eGlobal = seq - regre;
clear regre;
sSq = sum(eGlobal.^2,1);
sigma = sSq ./ (m-degreeOfFreedom-1); 
end

function [e] = computeSigmaIID(seq,G,beta)   
e = computeSSQSinglePx(seq,G,beta);
%regre = G*beta;
%eGlobal = seq - regre;
%clear regre;
%e = sum(eGlobal.^2,1);
end

function R = computeR(seq,SSQ)
    SStot = bsxfun(@minus,seq,mean(seq));
    SStot = sum(SStot.^2,1);
    R = 1 - SSQ ./ SStot;    
end

function [sSq] = computeSSQSinglePx(seq,G,beta)
[~,degreeOfFreedom] = size(G);
[m,n] = size(seq);
sSq(1:n) = 0;
for i=1:n
	y_i = seq(:,i);
    beta_i = beta(:,i);
    sSq(i) =  sum( (y_i - G*beta_i).^2 );
end

sSq = sSq ./ (m-degreeOfFreedom-1); 

end

function [err] = computeErrSinglePx(seq,G,beta)
[~,degreeOfFreedom] = size(G);
[m,n] = size(seq);
err(1:n,1:m) = single(0);
%bTg = beta'*G';
for i=1:n
    if(i == 89847)
        fprintf('');
    end
	y_i = seq(:,i);
    beta_i = beta(:,i);
    %btg_i = bTg(i,:);
	%sSq(i) =  (seq(:,i)'*seq(:,i) - bTg(i,:)*seq(:,i));  
    %%sSq(i) =  (y_i'*y_i - beta_i'*G'*y_i); 
    err(i,:) =  y_i - G*beta_i;
end

%err = err ./ (m-degreeOfFreedom-1); 

end


function [act] = computeSTEPActivityPattern(lag, phaseDuration, stim_width, t_start, timing)
% phaseDuration in s
% stim_width in s
% lag in s
act(1:length(timing)) = single(0);
i_beginStimulation = 1:2:20;
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;

for i=1:10
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2 > i1)
        act(i1:i2) = 1;
    end
end

end

function [act] = computeActivityPattern(lag, phaseDuration, stim_width, t_start, timing)
% phaseDuration in s
% stim_width in s
% lag in s
%act(1:length(timing),1:7) = single(0);
act = [];
i_beginStimulation = 1:2:20;
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;

% gaussian_glm(X,4,4) + gaussian_glm(X,8,4) + gaussian_glm(X,12,4) + gaussian_glm(X,16,4) + gaussian_glm(X,20,4) + gaussian_glm(X,24,4)+ 2*gaussian_glm(X,28,4)
for i=1:length(t_phaseShift)
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2 > i1)
        
        idx = (timing(i1:i2) - timing(i1)) ./ 1000;
        act(i1:i2) = gaussian_glm(idx) - min(gaussian_glm(idx));
        %act(i1:i2,j) = inverselogit(idx) - min(inverselogit(idx));
        % gaussian_glm(X,4,4) + gaussian_glm(X,8,4) + gaussian_glm(X,12,4) + gaussian_glm(X,16,4) + gaussian_glm(X,20,4) + gaussian_glm(X,24,4)+ gaussian_glm(X,28,4)
        
        
%         lAct(1:length(timing)) = 0;
%         lAct(i1:i2) = gaussian_glm(idx,4,4);
%         lAct(i1:i2) = lAct(i1:i2) ./ max(lAct(i1:i2));
%         act = [act lAct'];
%         
%         lAct(1:length(timing)) = 0;
%         lAct(i1:i2) = gaussian_glm(idx,8,4);
%         lAct(i1:i2) = lAct(i1:i2) ./ max(lAct(i1:i2));
%         act = [act lAct'];
%         
%         lAct(1:length(timing)) = 0;
%         lAct(i1:i2) = gaussian_glm(idx,12,4);
%         lAct(i1:i2) = lAct(i1:i2) ./ max(lAct(i1:i2));
%         act = [act lAct'];
%         
%         lAct(1:length(timing)) = 0;
%         lAct(i1:i2) = gaussian_glm(idx,16,4);
%         lAct(i1:i2) = lAct(i1:i2) ./ max(lAct(i1:i2));
%         act = [act lAct'];
%         
%         lAct(1:length(timing)) = 0;
%         lAct(i1:i2) = gaussian_glm(idx,20,4);
%         lAct(i1:i2) = lAct(i1:i2) ./ max(lAct(i1:i2));
%         act = [act lAct'];
%         
%         lAct(1:length(timing)) = 0;
%         lAct(i1:i2) = gaussian_glm(idx,24,4);
%         lAct(i1:i2) = lAct(i1:i2) ./ max(lAct(i1:i2));
%         act = [act lAct'];
%         
%         lAct(1:length(timing)) = 0;
%         lAct(i1:i2) = gaussian_glm(idx,28,4);
%         lAct(i1:i2) = lAct(i1:i2) ./ max(lAct(i1:i2));
%         act = [act lAct'];
    end
end

end


function [act] = computeBoxCarActivityPatternV1(lag, phaseDuration, stim_width, t_start, timing,widthBoxCar)
% phaseDuration in s
% stim_width in s
% lag in s
%act(1:length(timing),1:7) = single(0);
act = [];
i_beginStimulation = 1:2:20;
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;

% gaussian_glm(X,4,4) + gaussian_glm(X,8,4) + gaussian_glm(X,12,4) + gaussian_glm(X,16,4) + gaussian_glm(X,20,4) + gaussian_glm(X,24,4)+ 2*gaussian_glm(X,28,4)
for i=1:length(t_phaseShift)
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2 > i1)
        
        idx = (timing(i1:i2) - timing(i1)) ./ 1000;
        %act(i1:i2) = gaussian_glm(idx) - min(gaussian_glm(idx));
        %act(i1:i2,j) = inverselogit(idx) - min(inverselogit(idx));
        % gaussian_glm(X,4,4) + gaussian_glm(X,8,4) + gaussian_glm(X,12,4) + gaussian_glm(X,16,4) + gaussian_glm(X,20,4) + gaussian_glm(X,24,4)+ gaussian_glm(X,28,4)
        
        dt = 0;
        j = 1;
        while(dt <= 30)
            ib1 = dt;
            ib2 = min(ib1+widthBoxCar,30); 
            [~,ib1t] = min(abs(idx-ib1));
            [~,ib2t] = min(abs(idx-ib2));
            
            if( (idx(ib2t)-idx(ib1t)) < 4)
                break;
            end
            
            if(i==1)
                lAct(1:length(timing)) = 0;
                lAct(i1+ib1t:min(i1+ib2t,i2)) = 1;
                act = [act lAct'];
            else
                act(i1+ib1t:min(i1+ib2t,i2),j) = 1;            
            end
            if(length(idx) <= ib2t+1)
                break;
            end
            dt = idx(ib2t+1);
            j  = j + 1;
        end
        
    end
end

end


function [act] = computeBoxCarActivityPatternV2(lag, phaseDuration, stim_width, t_start, timing,widthBoxCar)
% phaseDuration in s
% stim_width in s
% lag in s
%act(1:length(timing),1:7) = single(0);
act = [];
i_beginStimulation = 1:2:20;
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;

% gaussian_glm(X,4,4) + gaussian_glm(X,8,4) + gaussian_glm(X,12,4) + gaussian_glm(X,16,4) + gaussian_glm(X,20,4) + gaussian_glm(X,24,4)+ 2*gaussian_glm(X,28,4)
for i=1:length(t_phaseShift)
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2 > i1)
        
        idx = (timing(i1:i2) - timing(i1)) ./ 1000;
        %act(i1:i2) = gaussian_glm(idx) - min(gaussian_glm(idx));
        %act(i1:i2,j) = inverselogit(idx) - min(inverselogit(idx));
        % gaussian_glm(X,4,4) + gaussian_glm(X,8,4) + gaussian_glm(X,12,4) + gaussian_glm(X,16,4) + gaussian_glm(X,20,4) + gaussian_glm(X,24,4)+ gaussian_glm(X,28,4)
        
        dt = 0;
        j = 1;
        while(dt < 30)
            ib1 = dt;
            ib2 = min(ib1+widthBoxCar,30); 
            [~,ib1t] = min(abs(idx-ib1));
            [~,ib2t] = min(abs(idx-ib2));
            
            %if(i==1)
                lAct(1:length(timing)) = 0;
                lAct(i1+ib1t:min(i1+ib2t,i2)) = 1;
                act = [act lAct'];
            %else
            %    act(i1+ib1t:min(i1+ib2t,i2),j) = 1;            
            %end
            if(length(idx) <= ib2t+1)
                break;
            end
            dt = idx(ib2t+1);
            j  = j + 1;
        end
        
    end
end

end

function [act] = computeBoxCarRestPattern(lag, phaseDuration, stim_width, t_start, timing,widthBoxCar)
% phaseDuration in s
% stim_width in s
% lag in s
%act(1:length(timing),1:7) = single(0);
act = [];
i_beginStimulation = 1:2:20;
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;

% gaussian_glm(X,4,4) + gaussian_glm(X,8,4) + gaussian_glm(X,12,4) + gaussian_glm(X,16,4) + gaussian_glm(X,20,4) + gaussian_glm(X,24,4)+ 2*gaussian_glm(X,28,4)
for i=1:length(t_phaseShift)
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2 > i1)
        
        idx = (timing(i1:i2) - timing(i1)) ./ 1000;
        %act(i1:i2) = gaussian_glm(idx) - min(gaussian_glm(idx));
        %act(i1:i2,j) = inverselogit(idx) - min(inverselogit(idx));
        % gaussian_glm(X,4,4) + gaussian_glm(X,8,4) + gaussian_glm(X,12,4) + gaussian_glm(X,16,4) + gaussian_glm(X,20,4) + gaussian_glm(X,24,4)+ gaussian_glm(X,28,4)
        
        dt = 0;
        j = 1;
        while(dt < 30)
            ib1 = dt;
            ib2 = min(ib1+widthBoxCar,60); 
            [~,ib1t] = min(abs(idx-ib1));
            [~,ib2t] = min(abs(idx-ib2));
            
            %if(i==1)
                lAct(1:length(timing)) = 0;
                lAct(i1+ib1t:min(i1+ib2t,i2)) = 1;
                act = [act lAct'];
            %else
            %    act(i1+ib1t:min(i1+ib2t,i2),j) = 1;            
            %end
            if(length(idx) <= ib2t+1)
                break;
            end
            dt = idx(ib2t+1);
            j  = j + 1;
        end
        
    end
end

end


function [act] = computeDiffActivityPattern(lag, phaseDuration, stim_width, t_start, timing)
% phaseDuration in s
% stim_width in s
% lag in s
act(1:length(timing)) = single(0);
i_beginStimulation = 1:2:20;
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;

for i=1:10
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2 > i1)
        %act(i1:i2) = 1;
        idx = (timing(i1:i2) - timing(i1)) ./ 1000;
        
        act(i1:i2-1) = diff(gaussian_glm(idx));
        act(i1:i2-1) = diff(inverselogit(idx));
        act(i1:i2-1) = act(i1:i2-1) ./ max(act(i1:i2-1));
    end
end

end

function [act] = computeDiff2ActivityPattern(lag, phaseDuration, stim_width, t_start, timing)
% phaseDuration in s
% stim_width in s
% lag in s
act(1:length(timing)) = single(0);
i_beginStimulation = 1:2:20;
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;

for i=1:10
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2 > i1)
        %act(i1:i2) = 1;
        idx = (timing(i1:i2) - timing(i1)) ./ 1000;
        act(i1:i2-2) = diff(gaussian_glm(idx),2);
        act(i1:i2-2) = diff(inverselogit(idx),2);
        act(i1:i2-2) = act(i1:i2-2) ./ max(act(i1:i2-2));
    end
end

end

function [act] = computeActivityPatternWithLinearIncreaseAndDecrease(lag, phaseDuration, stim_width, t_start, timing)
% phaseDuration in s
% stim_width in s
% lag in s
act(1:length(timing)) = single(-1);
i_beginStimulation = 1:2:20;
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;
t_rise = 5000; %ms

waitingTime = 5000; % ms
dt = timing(2)-timing(1); % lets hope there's no shutter in one of these two timepoints...
noIgnoreTempRiseFr = round(waitingTime/dt);
noRisingFr = round(t_rise/dt);

for i=1:10
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2 > i1)
        rise = @ (x) (2/noRisingFr)*x + (1 - 2*i1/noRisingFr);
        fall = @ (x) (-2/noRisingFr)*x + (1 + 2*i2/noRisingFr);
        %act(i1-noIgnoreTempRiseFr:i1) = 0;
        act(i1-noRisingFr : i1) = rise(i1-noRisingFr : i1);
        act(i1:i2) = 1;
        act(i2:min(i2+noRisingFr,length(timing))) = fall(i2:min(i2+noRisingFr,length(timing)));
        %act(i2-noIgnoreTempRiseFr:i2) = 0;
    end
end

end


function [act] = computeShutterPattern(T)
dT = diff(T);
t_S = find(dT > 4*mean(dT));
act(1:floor(length(t_S)/2)+1,1:length(T)) = 0;
if(numel(t_S) == 1)
    act(1:t_S(1)-1) = -1;
    act(t_S(1):end) = 1;
    return
end

act(1,1:t_S(1)-1) = -1;
act(1,t_S(1):t_S(2)) = 1;

for i=3:2:length(t_S)
    act(ceil(i/2),t_S(i-1):t_S(i)-1) = -1;
    if(i < length(t_S))
        act(ceil(i/2),t_S(i):t_S(i+1)) = 1;
    else
        act(ceil(i/2),t_S(i):end) = 1;
    end
end

if(mod(length(t_S),2) == 0)
    act(floor(length(t_S)/2)+1,t_S(end):end) = -1;
end

% i = length(t_S);
% if(i==3)
%     act(floor(length(t_S)/2)+1,t_S(i-1):t_S(i)-1) = -1;
%     act(floor(length(t_S)/2)+1,t_S(i):end) = 1;
% else
%     act(floor(length(t_S)/2)+1,t_S(i):end) = -1;
% end


end

function [act] = computeShutterPatternV2(T)
dT = diff(T);
t_S = find(dT > 4*mean(dT));
t_S = locateNUCpointsNew(dT)';
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

function [act] = computeMeanPattern(seq,T)
dMS = diff(mean(seq));
dT = diff(T);
t_S = find(dT > 4*mean(dT));
%t_S = locateNUCpointsNew2(dT,dMS,mean(seq))'; %auskommentiert 27.11.2014
t_S = locateNUCpointsNew(dT)';
if(t_S(1) == 1)
    t_S = t_S(2:end);
end
act(1:length(t_S)+1,1:length(T)) = 0;
mi = mean(seq(:,1:t_S(1)-1));
act(1,1:t_S(1)-1) = (mi - mean(mi(:))) ;

for i=2:length(t_S)
    mi = mean(seq(:,t_S(i-1):t_S(i)-1));
    act(i,t_S(i-1):t_S(i)-1) = (mi - mean(mi(:))) ;
end
mi = mean(seq(:,t_S(end):end));
act(length(t_S)+1,t_S(end):end) = (mi - mean(mi(:))) ;


end

function [act] = computeLinearPatternForNUCPeriods(seq,T)

dT = diff(T);
t_S = find(dT > 4*mean(dT));
t_S = locateNUCpointsNew(dT)';
if(t_S(1) == 1)
    t_S = t_S(2:end);
end
act(1:length(t_S)+1,1:length(T)) = 0;
act(1,1:t_S(1)-1) = 1:length(1:t_S(1)-1) ;

for i=2:length(t_S)
    act(i,t_S(i-1):t_S(i)-1) = 1:length(t_S(i-1):t_S(i)-1) ;
end

act(length(t_S)+1,t_S(end):end) =  1:length(t_S(end):length(T)) ;


end

function [act] = computePolynomialPatternForNUCPeriods(seq,T,p)

dT = diff(T);
t_S = find(dT > 4*mean(dT));
t_S = locateNUCpointsNew(dT)';
if(t_S(1) == 1)
    t_S = t_S(2:end);
end
act(1:length(t_S)+1,1:length(T)) = 0;
X = 1:length(1:t_S(1)-1);
X = X ./ X(end);
act(1,1:t_S(1)-1) = X.^p;

for i=2:length(t_S)
    X = 1:length(t_S(i-1):t_S(i)-1);
    X = X ./ X(end);
    act(i,t_S(i-1):t_S(i)-1) = X.^p;
end

X = 1:length(t_S(end):length(T));
X = X ./ X(end);
act(length(t_S)+1,t_S(end):end) = X.^p;


end

function [act] = computeVCPPatterns(seq,T,noComp)

dT = diff(T);
t_S = find(dT > 4*mean(dT));
t_S = locateNUCpointsNew(dT)';
if(t_S(1) == 1)
    t_S = t_S(2:end);
end
act(1:noComp*(length(t_S)+1),1:length(T)) = 0;
act(1:noComp,1:t_S(1)-1) = computeVCPs(seq(:,1:t_S(1)-1),noComp);

for i=2:length(t_S)
    act((i-1)*noComp + 1 : (i-1)*noComp + noComp ,t_S(i-1):t_S(i)-1) = computeVCPs(seq(:,t_S(i-1):t_S(i)-1),noComp);
end

act(i*noComp + 1 : end,t_S(end):end) = computeVCPs(seq(:,t_S(end):end),noComp);


        

end

function [Vi] = computeVCPs(seq,noExpectedDriftComponents)

[~,m] = size(seq);
        mu = sum(seq,2)/m;
        %xc = seq;
        xc = bsxfun(@minus,seq,mu);
        clear mu;
        [V, D] = eig(xc'*xc);
        D = abs(diag(D));
        clear xc;
        Vi = V(:,end-noExpectedDriftComponents+1:end)';
        fprintf('   explained variance: %f\n',sum(D(end-noExpectedDriftComponents+1:end)) / sum(D));

end
function [ t_minAICc, t2_minAICc, t_minGCV, t2_minGCV, minAICidx ] = GLMPSPLINEDWTwithDTAndSigmaopt( seq,T )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[noPixels,noTimepoints] = size(seq);
stim_width = 30;
t_start = T(1);
t_stim = 30;
%lag is initialized below

%% WAVELET STUFF
qmf = MakeONFilter('Symmlet',8);
L=1;
XI=eye(1024);
W=zeros(1024); % DWT BASIS
for i=1:1024
    W(:,i)=FWT_PO(XI(:,i),L,qmf);
end

%Wf(129:end,:) = 0; % no penalty on higher order frequency terms
% [stimPtnr2,stimPtnr_d] = computeGaussianActivityPatternV1(30+lag, t_stim, stim_width, t_start, T);
% [stimPtnr,stimPtnr_d] = computeGaussianActivityPatternV1(lag, t_stim, stim_width, t_start, T);


%% build design matrix
%fprintf('model %d\n',modelSelector);
[shutterPtnrV3,~,t_S] = computeShutterPatternV3(seq,T);
%BSpline basis
[B1,D1] = computePenBSplineBasis(noTimepoints,3,3,10);
[B2,D1] = computePenBSplineBasis(noTimepoints,3,3,400); % changed from 400 to 100 due to sigma_e testing

lag = 0;
lags = [linspace(0,7,8) 15];
sigmas = [linspace(4,10,3) 15];
%stimPtnr = computeBoxCarActivityPatternV1(lag,t_stim, stim_width, t_start, T,30-lag)';
stimPtnr = computeGaussianActivityPatternV2(lag, t_stim, stim_width, t_start, T);
G = [stimPtnr' shutterPtnrV3' B1];

lags = lags - 10;
for tLagStim=1:length(lags) % FIXME FIXME FIXME!!!!!!!!!!!!!!! tLagStim is an index not the actual time
    for si=1:length(sigmas)
    %[stimPtnr] = computeBoxCarActivityPatternV1(lag,t_stim, stim_width, t_start, T,30-lag)';
        stimPtnr = computeGaussianActivityPatternV2(tLagStim, t_stim, stim_width, t_start, T,sigmas(si));
        PRECOMP{si + (tLagStim-1)*length(sigmas)}.stimPtnr = stimPtnr;
  %  PRECOMP{tLagStim}.dStimPtnr = stimPtnr_d
    end
end



%G = [stimPtnr' shutterPtnrV3' B1];
[~,noFixedEffects] = size(G);
G = [G B2];
[~,noVariables] = size(G);
noPruningEffects = noVariables - noFixedEffects;


%% penalty matrix
%S = diag(ones(noPruningEffects,1),noFixedEffects);
S = eye(noVariables);
S(1:noFixedEffects,1:noFixedEffects) = 0;
P = eye(noTimepoints);
%P(256:end,256:end) = 0;
P(129:end,129:end) = 0;
%P(1:16,1:16) = 0;
%P(65:end,65:end) = 0;

Pterm = S'*G'*W'*P'*P*W*G*S;



%% initialize data for iterations
%lambdas = [0 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5];
lambdas = [linspace(0.001,0.1,30)  linspace(0.2,5,10)];

seq = seq';
%b = G'*seq;
t(1:length(lambdas),1:noPixels) = 0;
t2(1:length(lambdas),1:noPixels) = 0;

t(1:noPixels) = 0;
t2(1:noPixels) = 0;

noPrecompElem = length(PRECOMP);

GCVlambda(1:length(lambdas)*noPrecompElem,1:noPixels) = 0;
AIClambda(1:length(lambdas)*noPrecompElem,1:noPixels) = 0;
t(1:length(lambdas)*noPrecompElem,1:noPixels) = 0;
t2(1:length(lambdas)*noPrecompElem,1:noPixels) = 0;

%% PARAMETER ESTIMATION
fprintf('****************************\n');
fprintf('******   ESTIMATION   ******\n');
for tlagIdx=1:noPrecompElem
    fprintf(' %d / %d \n',tlagIdx,noPrecompElem);
    G(:,1) = PRECOMP{tlagIdx}.stimPtnr;
    GtG = G'*G;
    dtIdxOffset = (tlagIdx-1)*length(lambdas);
    for i=1:length(lambdas)
        lambda_i = lambdas(i);
%        fprintf('-> t %.1f l %.5f\n',lags(tlagIdx), lambda_i);
        GtGpD = (GtG + lambda_i .* Pterm);
        GTGpDsG = GtGpD\G';
        beta = GTGpDsG * seq;
        seqF = G*beta;
        eGlobal = seq - seqF;
        RSS = sum(eGlobal.^2,1);
        df = trace(GTGpDsG * G);
        AIClambda(i+dtIdxOffset,:) = log(RSS) + (2 * (df+1)) / (noTimepoints-df-2) ;
        GCVlambda(i+dtIdxOffset,:) = RSS ./ ((1-df/noTimepoints)^2);
        
        fprintf('-> µAIC %.10f   µGCV %.10f\n',mean(AIClambda(i+dtIdxOffset,:)),mean(GCVlambda(i+dtIdxOffset,:)));
        fprintf('\n');
        %StS = GTGpDsG' * GtG * GTGpDsG;
        %df_res = m - 2*df - trace(StS);
        %sigma_e = RSS ./ df_res;
        s_square = RSS ./ (noTimepoints-df-1);
        %s_square = sigma_e;
        covA = GtGpD\GtG/GtGpD;
        t(i+dtIdxOffset,:) = beta(1,:) ./ (sqrt(s_square .* covA(1,1)));
        t2(i+dtIdxOffset,:) = (beta(1,:)-beta(2,:)) ./ (sqrt(s_square .* abs(covA(1,1)-covA(2,2))));
        
        %tt = t(i,:);
        %tt(abs(tt) < 5) = NaN;
        %figure();imagesc(reshape(tt,480,640));title(sprintf('lag %d s lambda %.4f',lag,lambda_i));drawnow;
    end
end

[~,minAICidx] = min(AIClambda);
[~,minGCVidx] = min(GCVlambda);
t_minAICc(1:noPixels) = 0;
t2_minAICc(1:noPixels) = 0;
t_minGCV(1:noPixels) = 0;
t2_minGCV(1:noPixels) = 0;

for i=1:noPixels
    t_minAICc(i) = t(minAICidx(i),i);
    t2_minAICc(i) = t2(minAICidx(i),i);
    
    t_minGCV(i) = t(minGCVidx(i),i);
    t2_minGCV(i) = t2(minGCVidx(i),i);
    %minAICc(i) = AICclambda(minAICcidx(i));
    %bestres(:,i) = globalres(minAICcidx(i),:,i);
    %tBonf_best(i) = t_Bonf(minAICcidx(i));
    %activation(i) = abs(t_minAICc(i)) > tBonf_best(i);
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
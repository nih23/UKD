function [act,act_d] = computeGaussianActivityPatternV2(lag, phaseDuration, stim_width, t_start, timing, sigma)
% phaseDuration in s
% stim_width in s
% lag in s
%act(1:length(timing),1:7) = single(0);

if(~exist('mu','var'))
    mu = 15;
end

if(~exist('sigma','var'))
    sigma = 10;
end


act(1:length(timing)) = 0;
act_d(1:length(timing)) = 0;
i_beginStimulation = 1:2:20;
t_phaseShift = i_beginStimulation*phaseDuration*1000;
%t_phaseEndShift = t_phaseShift + stim_width*1000;

timing = (timing-timing(1)) ./ 1000;
for i=1:length(t_phaseShift)   
    mu = t_phaseShift(i)/1000 + lag;
    if(mu > timing(end))
       break;
    end
    [g,g_d] = gaussian_glm(timing,mu,sigma);
    
    [~,ir] = min(abs(timing - (mu+1.5*sigma))); % war vorher 2 (24.8.)
    [~,il] = min(abs(timing - (mu-1.5*sigma))); % war vorher 2 (24.8.)
    
    gaussianPattern = g;
    gaussianPattern(1:il) = nan;
    gaussianPattern(ir:end) = nan;
    gaussianPattern = gaussianPattern - min(gaussianPattern);
    gaussianPattern = gaussianPattern ./ max(gaussianPattern);
    gaussianPattern(isnan(gaussianPattern)) = 0;
    
    act = act + gaussianPattern;
    act_d = act_d + g_d;   
end

end




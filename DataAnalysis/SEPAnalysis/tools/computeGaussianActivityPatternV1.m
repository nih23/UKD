function [act,act_d] = computeGaussianActivityPatternV1(lag, phaseDuration, stim_width, t_start, timing, mu, sigma)
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
t_phaseShift = t_start + i_beginStimulation*phaseDuration*1000 + lag*1000;
t_phaseEndShift = t_phaseShift + stim_width*1000;
for i=1:length(t_phaseShift)
    [~, i1] = min(abs(timing - t_phaseShift(i)));
    [~, i2] = min(abs(timing - t_phaseEndShift(i)));
    if(i2-i1 < 5)
        continue;
    end
    idx = (timing(i1:i2) - timing(i1)) ./ 1000;
    
    [g,g_d] = gaussian_glm(idx,mu,sigma);
    
    gaussianPattern = g - min(g);
    gaussianPattern = gaussianPattern ./ max(gaussianPattern);
    act(i1:i2) = gaussianPattern;        
    
    act_d(i1:i2) = g_d;  
end



end


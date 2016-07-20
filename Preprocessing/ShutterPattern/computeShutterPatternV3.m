function [act, weights, t_S] = computeShutterPatternV3(seq,T)
ms = mean(seq);
dMS = diff(ms);
dT = diff(T);
t_S = find(dT > 510);
%fprintf('Shutter pattern > 510!\n');

%t_S = locateNUCpointsNew3(dT,dMS,mean(seq))'; %auskommentiert 27.11.2014
%t_S = locateNUCpointsNew(dT)';% auskommentiert 12.12.
n = length(t_S);

if(n == 1)
    weights(1:length(T)) = 1;
    act(1:n,1:length(T)) = -1;
    act(1,t_S(1):end) = 1;    
    return 
end

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

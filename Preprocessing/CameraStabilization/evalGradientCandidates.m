function [ candPlus, candMinus ] = evalGradientCandidates( sortedIndices,dx,dy,noEdges )
%EVALGRADIENTCANDIDATES This function estimates two groups. The first
%contains noEdges strongest values of sortedIndices. The second group
%ocntains of elements, that inherit the lowest values and are not in the
%neighbourhood of group 1.

%noEdges = 1000;
targetDist = 50;
candidatesPlus(1:2,1:noEdges) = 0;

for i=1:noEdges
    [j,k] = ind2sub([dx dy],sortedIndices(i));
    candidatesPlus(:,i) = [j k];
end

% candidatesPlus = [sortedIndices(1)];
% noFnd = 0;
% for i=2:length(sortedIndices)-1
%    [j,k] = ind2sub([dx dy],sortedIndices(i));
%    distToStrGrad = min(sum(abs(bsxfun(@minus,candidatesPlus,[j k]'))));
%    
%    if(distToStrGrad > targetDist)
%        candidatesPlus = [candidatesPlus sortedIndices(i)];
%        noFnd = noFnd + 1;
%    end
%    
%    if(noFnd == noEdges)
%        break;
%    end
% end
candPlus = candidatesPlus;
candMinus = [];
noFnd = 0;
for i=0:length(sortedIndices)-1
    [j,k] = ind2sub([dx dy],sortedIndices(end-i));
   distToStrGrad = min(sum(abs(bsxfun(@minus,candidatesPlus,[j k]'))));
   
   if(distToStrGrad > targetDist)
       candMinus = [candMinus sortedIndices(end-i)];
       noFnd = noFnd + 1;
   end
   
   if(noFnd == noEdges)
       break;
   end
end

candPlus = sortedIndices(1:noEdges);

end
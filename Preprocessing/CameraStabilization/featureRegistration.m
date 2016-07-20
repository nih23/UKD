function [ seqR ] = featureRegistration( seq, dx, dy )
% Summary of this function goes here
%   Detailed explanation goes here
% http://www.mathworks.de/de/help/vision/ug/feature-detection-extraction-and-matching.html#btj3wqb
[n,m] = size(seq);
seqR(1:n,1:m) = seq;

%fprintf('Camera motion correction using SURF-Feature matching\n');

imgA = mat2gray(reshape(seq(:,1),dx,dy));
pointsA = detectSURFFeatures(imgA,'MetricThreshold',4);
[featuresA, pointsA] = extractFeatures(imgA, pointsA);
%figure();
for i=2:m
   fprintf('%d/%d\n',i,m);
   try
   imgB = mat2gray(reshape(seq(:,i),dx,dy)); 
   imgA = mat2gray(reshape(seqR(:,i-1),dx,dy));
%     if(mod(i,100) == 0)
%         imgA = mat2gray(reshape(seq(:,i-1),dx,dy));
%         pointsA = detectSURFFeatures(imgA,'MetricThreshold',4);
%     end
   
    pointsB = detectSURFFeatures(imgB,'MetricThreshold',4);
    [featuresB, pointsB] = extractFeatures(imgB, pointsB);
    [featuresA, pointsA] = extractFeatures(imgA, pointsA);
    indexPairs = matchFeatures(featuresA, featuresB);
    pointsA = pointsA(indexPairs(:, 1), :);
    pointsB = pointsB(indexPairs(:, 2), :);
    [tform, pointsBm, pointsAm] = estimateGeometricTransform(pointsB, pointsA, 'similarity');
    imgBp = imwarp(reshape(seq(:,i),dx,dy), tform, 'OutputView', imref2d(size(imgB)),'interp','cubic');
 pointsBmp = transformPointsForward(tform, pointsBm.Location);

 showMatchedFeatures(reshape(seq(:,1),dx,dy), imgBp, pointsAm, pointsBmp);
 legend('A', 'B');drawnow;

    seqR(:,i) = imgBp(:);
   catch e
       fprintf('err\n');
   end 
end

end


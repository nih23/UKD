function [segmentedData] = cortexSegmentation(thermoData)
%% preprocessing
% create test data
%unaryTerms = randn(307200,2); % unary potentials
% project into wavelet domain
qmf = MakeONFilter('Symmlet',4);
L = 1;
[ seq_wc ] = FWT_PO_SEQUENCE( thermoData, L, qmf );
seq_wc_mean = [averageDyads(seq_wc) mean(thermoData,2)] ;
seq_wc_mean = [seq_wc(:,dyad(2)) seq_wc(:,dyad(3)) seq_wc(:,dyad(4)) seq_wc(:,dyad(5)) seq_wc(:,dyad(6)) mean(thermoData,2)] ;
seq_wc_mean = [seq_wc(:,dyad(5)) seq_wc(:,dyad(6)) mean(thermoData,2)] ;


%seq_wc_mean = [averageDyads(seq_wc)] ;

% extract some relevant components

%% SUPERVISED:
% LEARNING
% project stuff into bins
% train linear classifier (DT) some weight to discriminate classes
% APPLICATION
% apply linear classifier to cortex frequency feature vector and get some
% sort of confidence

%% UNSUPVERVISED
fprintf(' 1. train 2-GMM on each component of the data for normalization \n');
[noPixels,noFeatures] = size(seq_wc_mean);
unaryTerms(1:307200,2) = 0;

%unaryPots(1:307200,2,noFeatures) = 0;
% for i=1:noFeatures
%    si = seq_wc_mean(:,i); 
    si = seq_wc_mean;
    twogmm = gmdistribution.fit(si,2,'CovType','full');
    post = posterior(twogmm,si);
%    unaryPots(:,1,i) = post(:,1);
%    unaryPots(:,2,i) = post(:,2);
    
    unaryTerms(:,1) = post(:,1);
    unaryTerms(:,2) = post(:,2);
    
%    li = mean(twogmm.mu);
%    seq_wc_mean(:,i) = seq_wc_mean(:,i) ./ li;
% end

% unaryTerms = prod(unaryPots,3);
%% inference
% plug prepared data into opengm with 1st-order MRF given potts potentials
segmentedData = mexOpenGMInference(unaryTerms);

% maybe we need to add some sort of convex hull to the resulting data...
end

function [avged] = averageDyads(seq_wc) 
    [n,m] = size(seq_wc);
    maxDyad = log(1024) / log(2) - 1;
    avged = zeros(n,maxDyad);
    for i=1:maxDyad
       avged(:,i) = mean(seq_wc(:,dyad(i)),2);    
    end

end
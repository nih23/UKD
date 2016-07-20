function MeasurePerformanceOfCortexSegmentationV4(resultFilenameOffset, methods, featureExtractors, dx,dy,dt)
%
% Parameters:
% resultFilenameOffset
% methods                 \subseteq {'RF','RF_CRF','SVM'}
% featureExtractors       \subseteq {'PCA','DAE', 'SAE','kMeans'}

if(~exist('resultFilenameOffset','var'))
    resultFilenameOffset = '';
end


%% Parameters
%methods = {'RF'}; % 'RF_CRF','SVM', 'SQB'
%featureExtractors = {'PCA' }; % 'PCA','SAE','kMeans'
%minDyad = 3;
%maxDyad = 9;

%% load test and validation data
load('d:\Nico\CortexSegmentationData\brose.mat');
seq = seq(:,200:1024+199);
seq_test{1}.mu = computeNormalizedMean(seq);
seq_test{1}.bgFiltered = filterBackground(seq,dx,dy,dt);
labels_test{1} = labels(:);

seqall = seq_test{1}.bgFiltered;
muall = seq_test{1}.mu;
labelsall = labels(:);
load('d:\Nico\CortexSegmentationData\dorn.mat');
seq = seq(:,200:1024+199);
seq_test{2}.mu = computeNormalizedMean(seq);
seq_test{2}.bgFiltered = filterBackground(seq,dx,dy,dt);
labels_test{2} = labels(:);

seqall = [seqall; seq_test{2}.bgFiltered];
muall = [muall; seq_test{2}.mu];
labelsall = [labelsall; labels(:)];
load('d:\Nico\CortexSegmentationData\horn.mat');
seq = seq(:,200:1024+199);
seq_test{3}.mu = computeNormalizedMean(seq);
seq_test{3}.bgFiltered = filterBackground(seq,dx,dy,dt);
labels_test{3} = labels(:);

seqall = [seqall; seq_test{3}.bgFiltered];
muall = [muall; seq_test{3}.mu];
labelsall = [labelsall; labels(:)];


[seq_sample, idx] = datasample(seqall,300000,'Replace',false);
labels_sample = labelsall(idx);
mu_sample = muall(idx);
%idx2 = setdiff(1:921600,idx);
%sample_test = seqall(idx2,:);
%labels_test = labelsall(idx2);

clear seqall;
clear labelsall;

load('d:\Nico\CortexSegmentationData\brudzinski.mat');
seq = seq(:,120:119+1024);
seq_val{1}.mu = computeNormalizedMean(seq);
seq_val{1}.bgFiltered = filterBackground(seq,dx,dy,dt);
labels_val{1} = labels(:);

load('d:\Nico\CortexSegmentationData\krengel.mat');
seq = seq(:,120:119+1024);
seq_val{2}.mu = computeNormalizedMean(seq);
seq_val{2}.bgFiltered = filterBackground(seq,dx,dy,dt);
labels_val{2} = labels(:);

resultsFilename_test = strcat('D:\Nico\owncloud\Promotion\Auswertungen\CS\v4_test_',resultFilenameOffset,date,'.csv');
resultsFilename_val = strcat('D:\Nico\owncloud\Promotion\Auswertungen\CS\v4_val_',resultFilenameOffset,date,'.csv');
if(exist(resultsFilename_test) == 0)
    fid1 = fopen(resultsFilename_test,'a');
    fprintf(fid1,'Method;FeatureExtractor;PostProc;SizeDictionary;Dyads;µTP;sTP;µFP;sFP;µTN;sTN;µFN;sFN;µTPR;sTPR;µFPR;sFPR;µRecall;sRecall;µPrecision;sPrecision;µAccuracy;sAccuracy;RuntimeTraining;RuntimeClassification\n');
    fclose(fid1);
end

if(exist(resultsFilename_val) == 0)
    fid1 = fopen(resultsFilename_val,'a');
    fprintf(fid1,'Method;FeatureExtractor;PostProc;SizeDictionary;Dyads;µTP;sTP;µFP;sFP;µTN;sTN;µFN;sFN;µTPR;sTPR;µFPR;sFPR;µRecall;sRecall;µPrecision;sPrecision;µAccuracy;sAccuracy;RuntimeTraining;RuntimeClassification\n');
    fclose(fid1);
end

mu_1 = seq_val{1}.mu;
mu_2 = seq_val{2}.mu;

res_visual = strcat('D:\Nico\owncloud\Promotion\Auswertungen\CS\class_visual_raw.mat');
save(res_visual, 'mu_1', 'mu_2');

for i=1:length(methods)
    for j=1:length(featureExtractors)
        for dictSize = 10 : 10 : 50
            %dictSize = 30;
            minDyad = 1;
            maxDyad = 9;
            %for minDyad = 1 : 9
            %    for maxDyad = minDyad : 9
                    classifier = methods{i};
                    featureExtractor = featureExtractors{j};
                    mode = 'useTemperatureAsIndependentFeature';
                    fprintf('dict size %d\n',dictSize);
% % %                     %% train model to all DWT scales at once and include mean temperature as additional feature to classifier
% % %                     fprintf(strcat('\n\n**********************\n',classifier,'withTemp_',featureExtractor,'-all_',num2str(dictSize),'_',strcat(num2str(minDyad),'-',num2str(maxDyad)),'\n**********************\n'));
% % %                     useTemperatureFilenameFlag = '';
% % %                     c_allDyad = BoFW4(mode,dictSize,minDyad,maxDyad,classifier,featureExtractor,false); % no normalization
% % %                     tic;c_allDyad = c_allDyad.train(sample, labels_sample);rt_train=toc;
% % %                     fprintf('*** TEST DATASETS ***\n');
% % %                     computeAndStorePerformance(c_allDyad,seq_test,labels_test, resultsFilename_test, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad,  rt_train);
% % %                     fprintf('*** VALIDATION DATASETS ***\n');
% % %                     computeAndStorePerformance(c_allDyad,seq_val,labels_val, resultsFilename_val, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad,  rt_train);
                    
                    useTemperatureFilenameFlag = 'norm-';
                    c_allDyad = BoFW4(mode,dictSize,minDyad,maxDyad,classifier,featureExtractor,true); % no normalization
                    tic;c_allDyad = c_allDyad.train(mu_sample, seq_sample, labels_sample);rt_train=toc;
                    fprintf('*** TEST DATASETS ***\n');
                    computeAndStorePerformance(c_allDyad,seq_test,labels_test, resultsFilename_test, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad,  rt_train);
                    fprintf('*** VALIDATION DATASETS ***\n');
                    res = computeAndStorePerformance(c_allDyad,seq_val,labels_val, resultsFilename_val, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad,  rt_train, 1);
                    
                    
                    
                    res_visual = strcat('D:\Nico\owncloud\Promotion\Auswertungen\CS\class_visual_',classifier,'_',featureExtractor,'_',int2str(dictSize),'.mat');
                    save(res_visual, 'res');
                    
                    fprintf('');
% % %                     mode = 'useTemperatureAsIndependentFeature';
% % %                     fprintf(strcat('\n\n**********************\n',classifier,'withTemp_',featureExtractor,'-all_',num2str(dictSize),'_',strcat(num2str(minDyad),'-',num2str(maxDyad)),'\n**********************\n'));
% % %                     useTemperatureFilenameFlag = 'single-';
% % %                     c_allDyad = BoFW4_singleDim(mode,dictSize,minDyad,maxDyad,classifier,featureExtractor,false); % no normalization
% % %                     tic;c_allDyad = c_allDyad.train(sample, labels_sample);rt_train=toc;
% % %                     fprintf('*** TEST DATASETS ***\n');
% % %                     computeAndStorePerformance(c_allDyad,seq_test,labels_test, resultsFilename_test, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad,  rt_train);
% % %                     fprintf('*** VALIDATION DATASETS ***\n');
% % %                     computeAndStorePerformance(c_allDyad,seq_val,labels_val, resultsFilename_val, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad,  rt_train);
                    
% % %                     useTemperatureFilenameFlag = 'single-norm-';
% % %                     c_allDyad = BoFW4_singleDim(mode,dictSize,minDyad,maxDyad,classifier,featureExtractor,true); % no normalization
% % %                     tic;c_allDyad = c_allDyad.train(sample, labels_sample);rt_train=toc;
% % %                     fprintf('*** TEST DATASETS ***\n');
% % %                     computeAndStorePerformance(c_allDyad,seq_test,labels_test, resultsFilename_test, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad,  rt_train);
% % %                     fprintf('*** VALIDATION DATASETS ***\n');
% % %                     computeAndStorePerformance(c_allDyad,seq_val,labels_val, resultsFilename_val, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad,  rt_train);
               % end
            %end
        end
    end
end
end

function [estLabelsRet] = computeAndStorePerformance(trainedClassifier,test_data,test_labels,resultsFilename, classifier, featureExtractor, useTemperatureFilenameFlag, dictSize, minDyad, maxDyad, rt_train, dispRes)
%% compute performance => All Dyads
fprintf('Classifying datasets [');
estLabelsRet = zeros(length(test_data), 640*480);
for k=1:length(test_data)
    fprintf('.');
    tic;[estLabels,score,estLabelsFilledHoles,estLabelsCH] = trainedClassifier.classify(test_data{k});rt_class=toc;
    %if(exist('dispRes','var'))
    %    figure();imshowpair(reshape(mat2gray(test_data{k}.mu), 480, 640), reshape(estLabels,480,640),'montage');drawnow;
    %end
    estLabelsRet(k,:) = estLabels(:);
    [ noTP(k),noFP(k),noTN(k),noFN(k),tpr(k),fpr(k),recall(k),precision(k),ACC(k),MCC(k) ] = computePerformanceOfGLMresult(test_labels{k},estLabels(:));
    [ noTP_fh(k),noFP_fh(k),noTN_fh(k),noFN_fh(k),tpr_fh(k),fpr_fh(k),recall_fh(k),precision_fh(k),ACC_fh(k),MCC_fh(k) ] = computePerformanceOfGLMresult(test_labels{k},estLabelsFilledHoles(:));
    [ noTP_cvx(k),noFP_cvx(k),noTN_cvx(k),noFN_cvx(k),tpr_cvx(k),fpr_cvx(k),recall_cvx(k),precision_cvx(k),ACC_cvx(k),MCC_cvx(k) ] = computePerformanceOfGLMresult(test_labels{k},estLabelsCH(:));
end
fprintf(']\n');

fid1 = fopen(resultsFilename,'a');
% no postproc
fprintf(fid1,'%s;%s;%s;%.3f;%s;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.3f;%.3f\n',classifier,strcat(featureExtractor,'-all-',useTemperatureFilenameFlag),'none',dictSize, strcat(num2str(minDyad),'->',num2str(maxDyad)),mean(noTP),std(noTP),mean(noFP),std(noFP),mean(noTN),std(noTN),mean(noFN),std(noFN),mean(tpr),std(tpr),mean(fpr),std(fpr),mean(recall),std(recall),mean(precision),std(precision),mean(ACC),std(ACC),rt_train,rt_class);
% filled holes
%fprintf(fid1,'%s;%s;%s;%.3f;%s;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.3f;%.3f\n',classifier,strcat(featureExtractor,'-all-',useTemperatureFilenameFlag),'fh',dictSize, strcat(num2str(minDyad),'->',num2str(maxDyad)),mean(noTP_fh),std(noTP_fh),mean(noFP_fh),std(noFP_fh),mean(noTN_fh),std(noTN_fh),mean(noFN_fh),std(noFN_fh),mean(tpr_fh),std(tpr_fh),mean(fpr_fh),std(fpr_fh),mean(recall_fh),std(recall_fh),mean(precision_fh),std(precision_fh),mean(ACC_fh),std(ACC_fh),rt_train,rt_class);
% convex hull
%fprintf(fid1,'%s;%s;%s;%.3f;%s;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.3f;%.3f\n',classifier,strcat(featureExtractor,'-all-',useTemperatureFilenameFlag),'cvx',dictSize, strcat(num2str(minDyad),'->',num2str(maxDyad)),mean(noTP_cvx),std(noTP_cvx),mean(noFP_cvx),std(noFP_cvx),mean(noTN_cvx),std(noTN_cvx),mean(noFN_cvx),std(noFN_cvx),mean(tpr_cvx),std(tpr_cvx),mean(fpr_cvx),std(fpr_cvx),mean(recall_cvx),std(recall_cvx),mean(precision_cvx),std(precision_cvx),mean(ACC_cvx),std(ACC_cvx),rt_train,rt_class);
fclose(fid1);


fprintf('=> prediction ACC %.3f TPR %.3f FPR %.3f TP %.1f FP %.1f TN %.1f FN %.1f\n',mean(ACC),mean(tpr),mean(fpr),mean(noTP),mean(noFP),mean(noTN),mean(noFN));
%fprintf('=> filled holes   ACC %.3f TPR %.3f FPR %.3f TP %.1f FP %.1f TN %.1f FN %.1f\n',mean(ACC_fh),mean(tpr_fh),mean(fpr_fh),mean(noTP_fh),mean(noFP_fh),mean(noTN_fh),mean(noFN_fh));
%fprintf('=> convex hull    ACC %.3f TPR %.3f FPR %.3f TP %.1f FP %.1f TN %.1f FN %.1f\n',mean(ACC_cvx),mean(tpr_cvx),mean(fpr_cvx),mean(noTP_cvx),mean(noFP_cvx),mean(noTN_cvx),mean(noFN_cvx));

fprintf('');
end

function [mu] = computeNormalizedMean(seq)
            mu = double(mean(seq,2));
            mu = mu - min(mu);
            mu = mu ./ max(mu);
end
        
function [seq_filt] = filterBackground(seq,dx,dy,dt)
    [seqn, rss] = SplineSmoothing3dImgsequenceNoNUC(seq,dx,dy,dt); 
    seqn = full(seqn);
    seq_filt = seq - seqn;
end

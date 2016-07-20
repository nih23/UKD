function [  ] = leistungsEvaluation( s1, gamma )
%EVAL Summary of this function goes here
%   Detailed explanation goes here

[n,m] = size(s1);
fprintf('\n*** \n CMC w/ gamma %f\n***\n',gamma);

signalLength = 2048;
maxDisplacement = 2*gamma+1;
maxDPdn = length(maxDisplacement:n-maxDisplacement);
maxDPdm = length(maxDisplacement:m-maxDisplacement);

maxDPdnMSE = length(2*maxDisplacement:n-2*maxDisplacement)+1;
maxDPdmMSE = length(2*maxDisplacement:m-2*maxDisplacement)+1;

%fprintf('displacement X %d Y %d\n',maxDPdn,maxDPdm);

s1real = repmat(s1(:),1,signalLength);
s1real = reshape(s1real,480,640,signalLength);
s1realC = s1real(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
s1realC = s1realC(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
s1realC = reshape(s1realC,maxDPdnMSE*maxDPdmMSE,signalLength);
clear s1real;

s1cmc = cmcSimulation(s1,20,gamma,104);
s1cmc = s1cmc(:,1:signalLength);
s1cmcC = reshape(s1cmc,480,640,signalLength);
s1cmcCMSE = s1cmcC(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
s1cmcCMSE = s1cmcCMSE(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
s1cmcCMSE = reshape(s1cmcCMSE,maxDPdnMSE*maxDPdmMSE,signalLength);
s1cmc=s1cmcC(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
s1cmc = reshape(s1cmc,maxDPdn*maxDPdm,signalLength);
%clear s1cmc;
clear s1cmcCMSE;
clear s1cmcC;

F = fft(eye(signalLength));
Fs = 1000 / 20;
NFFT = 2^nextpow2(signalLength);
f = Fs/2*linspace(0,1,NFFT/2+1);
[~,lIdx] = min(abs(f-2));
[~,rIdx] = min(abs(f-10));

s1cmc = double(s1cmc);

% % % Y = F*s1cmcCMSE';
% % % Y = Y(1:NFFT/2+1,:);
% % % IFSraw = (sum(sum(abs(Y(lIdx:rIdx,:)).^2))) / (maxDPdmMSE*maxDPdnMSE);
% % % %IFSraw = abs(sum(sum(Y(lIdx:rIdx,:)))) /  (maxDPdmMSE*maxDPdnMSE);
% % % mseraw = sqrt(sum(sum((s1realC-s1cmcCMSE).^2)) /  (maxDPdmMSE*maxDPdnMSE*signalLength));
% % % %psnrraw = 20*log(max(max(max(s1cmcCMSE)))) - 10*log(mseraw);
% % % fprintf('RAW RMSE %f IFS %f\n',mseraw, IFSraw);
% % % clear Y;
% % % clear s1cmcC;
% % % clear s1cmcCMSE;
% % % 

% % % tic;s1cmcEigenFeat = featureRegistration(s1cmc,maxDPdn,maxDPdm);timeFeat=toc;
% % % s1cmcEigenFeat = reshape(s1cmcEigenFeat,maxDPdn,maxDPdm,signalLength);
% % % s1cmcEigenFeat = s1cmcEigenFeat(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
% % % s1cmcEigenFeat = reshape(s1cmcEigenFeat,maxDPdmMSE*maxDPdnMSE,signalLength);
% % % mseFeat = sqrt(sum(sum((s1realC-s1cmcEigenFeat).^2)) / (maxDPdmMSE*maxDPdnMSE*signalLength));
% % % Y = F*s1cmcEigenFeat';
% % % IFSfeat = (sum(sum(abs(Y(lIdx:rIdx,:)).^2))) / (maxDPdmMSE*maxDPdnMSE);
% % % %IFSfeat = abs(sum(sum(Y(lIdx:rIdx,:)))) / (maxDPdmMSE*maxDPdnMSE);
% % % psnrfeat = 20*log(max(max(max(s1cmcEigenFeat)))) - 10*log(mseFeat);
% % % fprintf('SURF RMSE %f IFS %f t %f\n',mseFeat,IFSfeat,timeFeat);
% % % clear Y;
% % % clear s1cmcEigenFeat;
% % % 
% % % % 
% % % tic;s1cmcvcpDBS = varianceComponentPruningForCMC(s1cmc,50,0,maxDPdn,maxDPdm);timeDBS=toc;
% % % s1cmcvcpDBS = reshape(s1cmcvcpDBS,maxDPdn,maxDPdm,signalLength);
% % % s1cmcvcpDBS = s1cmcvcpDBS(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
% % % s1cmcvcpDBS = reshape(s1cmcvcpDBS,maxDPdmMSE*maxDPdnMSE,signalLength);
% % % mseDBS = sqrt(sum(sum((s1realC-s1cmcvcpDBS).^2)) / (maxDPdmMSE*maxDPdnMSE*signalLength));
% % % Y = F*s1cmcvcpDBS';
% % % Y = Y(1:NFFT/2+1,:);
% % % IFSDBS = (sum(sum(abs(Y(lIdx:rIdx,:)).^2))) / (maxDPdmMSE*maxDPdnMSE);
% % % psnrvcpdbs = 20*log(max(max(max(s1cmcvcpDBS)))) - 10*log(mseDBS);
% % % fprintf('vcpNP RMSE %f IFS %f t %f\n',mseDBS,IFSDBS,timeDBS);
% % % clear Y;
% % % clear s1cmcvcpDBS;
% % % 

tic;s1cmcvcp1SVM = varianceComponentPruningForCMC(s1cmc,50,1,maxDPdn,maxDPdm);time1SVM=toc;
s1cmcvcp1SVM = reshape(s1cmcvcp1SVM,maxDPdn,maxDPdm,signalLength);
s1cmcvcp1SVM = s1cmcvcp1SVM(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
s1cmcvcp1SVM = reshape(s1cmcvcp1SVM,maxDPdmMSE*maxDPdnMSE,signalLength);
mse1SVM = sqrt(sum(sum((s1realC-s1cmcvcp1SVM).^2)) / (maxDPdmMSE*maxDPdnMSE*signalLength));
Y = F*s1cmcvcp1SVM';
clear s1cmcvcp1SVM;
Y = Y(1:NFFT/2+1,:);
IFS1SVM = (sum(sum(abs(Y(lIdx:rIdx,:)).^2))) / (maxDPdmMSE*maxDPdnMSE);
%psnrvcp1svm = 20*log(max(max(max(s1cmcvcp1SVM)))) - 10*log(mse1SVM);
fprintf('vcp1SVM RMSE %f IFS %f t %f\n',mse1SVM,IFS1SVM,time1SVM);
% fprintf('vcp1SVM RMSE %f t %f\n',mse1SVM,time1SVM);
% % % clear Y;
% % % clear s1cmcvcp1SVM;


tic;s1cmcvcp1SVM = varianceComponentPruningForCMC(s1cmc,50,2,maxDPdn,maxDPdm);time1SVM=toc;
s1cmcvcp1SVM = reshape(s1cmcvcp1SVM,maxDPdn,maxDPdm,signalLength);
s1cmcvcp1SVM = s1cmcvcp1SVM(maxDisplacement:end-maxDisplacement,maxDisplacement:end-maxDisplacement,:);
s1cmcvcp1SVM = reshape(s1cmcvcp1SVM,maxDPdmMSE*maxDPdnMSE,signalLength);
mse1SVM = sqrt(sum(sum((s1realC-s1cmcvcp1SVM).^2)) / (maxDPdmMSE*maxDPdnMSE*signalLength));
Y = F*s1cmcvcp1SVM';
clear s1cmcvcp1SVM;
Y = Y(1:NFFT/2+1,:);
IFS1SVM = (sum(sum(abs(Y(lIdx:rIdx,:)).^2))) / (maxDPdmMSE*maxDPdnMSE);
%psnrvcp1svm = 20*log(max(max(max(s1cmcvcp1SVM)))) - 10*log(mse1SVM);
% fprintf('vcpGT RMSE %f t %f\n',mse1SVM,time1SVM);
clear Y;




end


classdef BoFW2MetaFunctionality
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        centroids
        svm
        k
        qmf
        L
        maxDyad
        minDyad
        minOffset
        classifier
        featureExtractor
        nTrees
        myIndices
        saeStackSize
    end
    
    methods
        
        function obj = BoFW2MetaFunctionality(noWords, minDyad, maxDyad, classifier, featureExtractor)
            obj.k = noWords;
            obj.maxDyad = maxDyad;
            obj.minDyad = minDyad;
            lminOffset = dyad(obj.minDyad);
            lminOffset = lminOffset(1) - 1;
            obj.minOffset = lminOffset;
            obj.classifier = classifier;
            obj.featureExtractor = featureExtractor;
            obj.myIndices = [];
            for di = obj.minDyad : obj.maxDyad
                obj.myIndices = [obj.myIndices dyad(di)];
            end
            
            %% constants
            obj.qmf = MakeONFilter('Symmlet',8);
            obj.L = 1;
            obj.saeStackSize = 3;
            obj.nTrees = 20;
        end
        
        function [labels,score, labels2, labels_ch] = applyClassifier(obj, model, proj)
            %% apply learnt model
            if (strcmp(obj.classifier,'SQB') == 1)
                score = SQBMatrixPredict( model, single(proj) );
                labels(1:307200) = 0;
                labels(score > 0) = 1;
            else
                [labels,score] = predict(model, proj);
            end
            
            if (strcmp(obj.classifier,'RF_CRF') == 1)
                % plug probabilities of RF as unary term into CRF model
                % with potts function as pairwise term and infer
                % new segmentation using TRW-S (GraphCut should work as
                % well..)
                C = 1;
                labels = mexOpenGMInference(fliplr(score), C);
            end
            
            if (strcmp(obj.classifier,'RF_CRF_N2') == 1)
                % plug probabilities of RF as unary term into CRF model
                % with potts function as pairwise term and infer
                % new segmentation using TRW-S (GraphCut should work as
                % well..)
                C = 1;
                labels = mexOpenGMInference_N2(fliplr(score), C);
            end
            
            if (strcmp(obj.classifier,'RF') == 1)
                labels = str2num(cell2mat(labels));
            end
            
            labels2 = postprocessSegmentation(obj, labels);
            
            labels_ch = bwconvhull(labels2);
        end
        
        function [ res ] = postprocessSegmentation(obj,  bw )
            noPx = length(bw(:));
            bw = reshape(bw,480,640);
            CC = bwconncomp(bw,4);
            n = CC.NumObjects;
            maxIdx = 1;
            for i=2:n
                if(length(CC.PixelIdxList{i}) > length(CC.PixelIdxList{maxIdx}))
                    maxIdx = i;
                end
            end
            
            res(1:noPx) = 0;
            if(CC.NumObjects > 0)
                res(CC.PixelIdxList{maxIdx}) = 1;
                res = imfill(reshape(res,480,640),'holes');
            end
        end
        
        function [model] = trainClassifier(obj, proj, labels)
            model = [];
            
            if(strcmp(obj.classifier,'SVM') == 1)
                %obj.svm = fitcsvm(proj, labels,'KernelFunction','rbf', 'KernelScale','auto','Standardize',true);
                model = fitcsvm(proj, labels,'KernelScale','auto','Standardize',true);
            elseif ( (strcmp(obj.classifier,'RF') == 1) || strcmp(obj.classifier,'RF_CRF') == 1 || strcmp(obj.classifier,'RF_CRF_N2') == 1)
                stream = RandStream('mlfg6331_64','Seed',23425);  % Random number stream
                options = statset('UseSubstreams',1,'Streams',stream);
                fprintf('--> %d trees\n',obj.nTrees);
                model = TreeBagger(obj.nTrees,proj,labels, 'Method', 'classification','options',options);   
            elseif ( (strcmp(obj.classifier,'SQB') == 1) )
                labels(labels == 0) = -1;
                % gradient boost options
                opts.loss = 'exploss';
                opts.shrinkageFactor = 0.1;
                opts.subsamplingFactor = 0.5;
                opts.maxTreeDepth = uint32(4);  % this was the default before customization
                opts.randSeed = uint32(23425);
                opts.mtry = uint32(ceil(sqrt(size(proj,2))));
                model = SQBMatrixTrain( single(proj), labels, uint32(200), opts );
            end
        end
        
        function [ci_temp] = learnFeatures(obj, featureExtractorName, seqWT)
            ci_temp{1} = [];
            if(strcmp(featureExtractorName,'vl_kMeans') == 1)
                vl_twister('state',2342);
                ci_temp{1} = vl_kmeans(seqWT', obj.k)'; 
            elseif(strcmp(featureExtractorName,'kMeans') == 1)
                stream = RandStream('mlfg6331_64','Seed',2342);  % Random number stream
                %1.3.2016 -> auskommentiert options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
                options = statset('UseSubstreams',1,'Streams',stream);
                [~,ci_temp{1}] = kmeans(seqWT, obj.k, 'Options',options,'Replicates',1); % 01.03.2016: replicates set to 1 for increased speed
            elseif(strcmp(featureExtractorName,'PCA') == 1)
                ci_temp{1} = obj.lpca(seqWT);
            elseif(strcmp(featureExtractorName,'DAE') == 1)
                ci_temp{1} = trainAutoencoder(seqWT', obj.k, 'MaxEpochs',2500);
            elseif(strcmp(featureExtractorName,'DAEGPU') == 1)
                ci_temp{1} = trainAutoencoder(double(seqWT(1:8:end,:)'), obj.k, 'MaxEpochs',2500, 'UseGPU',true);
            elseif(strcmp(featureExtractorName,'SAE') == 1)
                % stacked auto encoder
                ci_temp{1} = obj.trainStackedAutoencoder(seqWT');
            elseif(strcmp(featureExtractorName,'SAEGPU') == 1)
                % stacked auto encoder
                ci_temp{1} = obj.trainStackedAutoencoderGPU(seqWT');
            elseif(strcmp(featureExtractorName,'fSAE') == 1)
                % stacked auto encoder
                ci_temp{1} = obj.trainFractionalStackedAutoencoder(seqWT');
            end
            
        end
        
         function [ci_temp] = learnDualFeatures(obj, featureExtractorName, featC1, featC2)
            ci_temp{1} = [];
            ci_temp{2} = [];
            if(strcmp(featureExtractorName,'vl_kMeans') == 1)
                vl_twister('state',2342);
                ci_temp{1} = vl_kmeans(featC1', obj.k)';
                ci_temp{2} = vl_kmeans(featC2', obj.k)';
            elseif(strcmp(featureExtractorName,'kMeans') == 1)
                stream = RandStream('mlfg6331_64','Seed',2342);  % Random number stream
                options = statset('UseSubstreams',1,'Streams',stream);
                [~,ci_temp{1}] = kmeans(featC1, obj.k, 'Options',options,'Replicates',1); % 01.03.2016: replicates set to 1 for increased speed
                [~,ci_temp{2}] = kmeans(featC2, obj.k, 'Options',options,'Replicates',1); % 01.03.2016: replicates set to 1 for increased speed
            elseif(strcmp(featureExtractorName,'PCA') == 1)
                ci_temp{1} = obj.lpca(featC1);
                ci_temp{2} = obj.lpca(featC2);
            elseif(strcmp(featureExtractorName,'DAE') == 1)
                ci_temp{1} = trainAutoencoder(featC1', obj.k, 'MaxEpochs',250);
                ci_temp{2} = trainAutoencoder(featC2', obj.k, 'MaxEpochs',250);
            elseif(strcmp(featureExtractorName,'DAEGPU') == 1)
                ci_temp{1} = trainAutoencoder(double(featC1(1:8:end,:)'), obj.k, 'MaxEpochs',250, 'UseGPU',true);
                ci_temp{2} = trainAutoencoder(double(featC2(1:8:end,:)'), obj.k, 'MaxEpochs',250, 'UseGPU',true);
            elseif(strcmp(featureExtractorName,'SAE') == 1)
                % stacked auto encoder
                ci_temp{1} = obj.trainStackedAutoencoder(featC1');
                ci_temp{2} = obj.trainStackedAutoencoder(featC2');
            elseif(strcmp(featureExtractorName,'SAEGPU') == 1)
                % stacked auto encoder
                ci_temp{1} = obj.trainStackedAutoencoderGPU(featC1');
                ci_temp{2} = obj.trainStackedAutoencoderGPU(featC2');
            elseif(strcmp(featureExtractorName,'fSAE') == 1)
                % stacked auto encoder
                ci_temp{1} = obj.trainFractionalStackedAutoencoder(featC1');
                ci_temp{2} = obj.trainFractionalStackedAutoencoder(featC2');
            end
            
        end
        
        %% fractional stacked AE training
        function [ sae_model ] = trainFractionalStackedAutoencoder(obj, f_i)
            fprintf(' train SAE layer %d\n', 1);
            sae_model{1} = trainAutoencoder(f_i, obj.k, 'MaxEpochs',250);
            
            for i = 2 : obj.saeStackSize
                fprintf(' train SAE layer %d\n', i);
                f_i = encode(sae_model{i-1}, f_i);
                sae_model{i} = trainAutoencoder(f_i, obj.k / 2^(i-1), 'MaxEpochs',250);
            end
            
        end
        
        %% stacked AE GPU training
        function [ sae_model ] = trainStackedAutoencoderGPU(obj, f_i)
            fprintf(' train SAE layer %d (GPU)\n', 1);
            sae_model{1} = trainAutoencoder(double(f_i(:,1:8:end)), obj.k, 'MaxEpochs',250, 'UseGPU',true);
            for i = 2 : obj.saeStackSize
                fprintf(' train SAE layer %d (GPU)\n', i);
                f_i = encode(sae_model{i-1}, f_i);
                sae_model{i} = trainAutoencoder(double(f_i(:,1:8:end)), obj.k, 'MaxEpochs',250, 'UseGPU',true);
            end
            
        end
        
        %% stacked AE GPU training
        function [ sae_model ] = trainStackedAutoencoder(obj, f_i, labels)
            fprintf(' train SAE layer %d\n', 1);
            sae_model{1} = trainAutoencoder(f_i, obj.k, 'MaxEpochs',250);
            
            for i = 2 : obj.saeStackSize
                fprintf(' train SAE layer %d\n', i);
                f_i = encode(sae_model{i-1}, f_i);
                sae_model{i} = trainAutoencoder(f_i, obj.k, 'MaxEpochs',250);
            end
            
            % soft max
            
            % stack all          
            stackednet = stack(sae_model{1},sae_model{2},sae_model{3}); % TODO: generalize this code to other number of AE layer
            % re-train net
            
            % activations -> trainFeatures = activations(net,XTrain,6);

        end
        
        %% stacked AE encoding
        function [f_i] = applySAE(obj, f_i, sae_model)
            for i = 1 : obj.saeStackSize
                f_i = encode(sae_model{i}, f_i);
            end
        end
        
        %%
        function scores = lpca_forward(obj, S, cj)
            scores = S * cj;
            scores = scores(:,max(end - obj.k + 1,1)    :    end   );
        end
        
        %%
        function Vl = lpca(obj, S)
            [Vl, ~] = eig(S'*S);
        end
        
        %%
        function [ projectedData ] = normalizeDyads(obj, seqWT)
            [n,~] = size(seqWT);
            projectedData = seqWT(:,1:2);
            for j = 1 : obj.maxDyad
                swt = zscore(seqWT(:,dyad(j)), [], 2);
                %swt = seqWT(:,dyad(j));
                projectedData = [projectedData swt];
            end
        end
        
        %%
        function [ projectedData ] = projectData(obj, seqWT, ci)
            if(strcmp(obj.featureExtractor,'none') == 1)
                projectedData = [];
                return;                
            end
            
            [n,~] = size(seqWT);
            
            projectedData = [];
            swt = seqWT(:,:);
            
            for j=1:length(ci) % ci can be of size > 1 in case of dual, triple, ... feature learning... :)
                cj = ci{j};

                if( (strcmp(obj.featureExtractor,'kMeans') == 1) || (strcmp(obj.featureExtractor,'vl_kMeans') == 1) )
                    %[proj,D] = knnsearch(cj,swt);
                    [proj,D] = knnsearch(cj,swt,'K',obj.k);
                    %projectedData(:,1) = knnsearch(cj,swt); % we might prolly be more interested to probability of belonging to each term..
                    projectedData = [projectedData proj]; %D entfernt. 1003. meta gute performance! -> just D also yielded good Performance!
                    %D2 = pdist2(swt,cj);
                    %projectedData = [projectedData D2];
                elseif(strcmp(obj.featureExtractor,'PCA') == 1)
                    proj = obj.lpca_forward(swt,cj); % applied to wrong dimension? transpose first?
                    %[proj,D] = knnsearch(cj,swt,'K',obj.k);
                    projectedData = [projectedData proj];
                elseif(  (strcmp(obj.featureExtractor,'DAE') == 1) || (strcmp(obj.featureExtractor,'DAEGPU') == 1))
                    proj = encode(cj,swt')';
                    projectedData = [projectedData proj];
                elseif( (strcmp(obj.featureExtractor,'SAE') == 1) || (strcmp(obj.featureExtractor,'SAEGPU') == 1) || (strcmp(obj.featureExtractor,'fSAE') == 1) )
                    proj = obj.applySAE(swt', cj)';
                    projectedData = [projectedData proj];
                end
            
            end
        end
        
    end
    
end


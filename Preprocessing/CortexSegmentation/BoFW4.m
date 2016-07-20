classdef BoFW4 < BoFW2MetaFunctionality
    
    properties
        mode  
        doNormalizeDyads
    end
    
    methods
        
        function obj = BoFW4(mode, noWords, minDyad, maxDyad, classifier, featureExtractor, normalizeDyads)
            obj = obj@BoFW2MetaFunctionality(noWords, minDyad, maxDyad, classifier, featureExtractor);
            obj.mode = mode;
            if(~exist('normalizeDyads','var'))
                normalizeDyads = false;
            end
            obj.doNormalizeDyads = normalizeDyads;
        end
        
        %%
        function obj = train( obj, mu, seq, labels, biFeatureTraining )
            if(~exist('biFeatureTraining','var'))
                biFeatureTraining = 0;
            end
            
            fprintf(strcat(obj.featureExtractor,'-',obj.classifier,' training phase started\n'));
            
            seqWT = single(FWT_PO_SEQUENCE(seq,obj.L,obj.qmf));
            if(obj.doNormalizeDyads == true)
                seqWT = normalizeDyads(obj, seqWT); % evt. Normalisierung komplett weglassen. Scales sind schon mean centered, std. evt. aussagekräftig!
            end
            seqWT = seqWT(:,obj.myIndices);

            %% find frequency bins specific for each dyad

            fprintf('[1/3]');
            if(biFeatureTraining == 1)
            	fprintf(' Bi');
            end
            fprintf(' Feature Learning\n');
            %% learn features
            if(biFeatureTraining == 0)
                obj.centroids{1} = obj.learnFeatures(obj.featureExtractor, seqWT(labels == 0,:));
                %obj.centroids{1} = obj.learnFeatures(obj.featureExtractor, seqWT(labels == 0,:));

            else
                obj.centroids{1} = obj.learnDualFeatures(obj.featureExtractor, seqWT(labels == 0,:), seqWT(labels == 1,:));
            end
            
            %% project data
            fprintf('[2/3] Data Projection\n');
            proj = projectData(obj, seqWT);
            clear seqWT;
            
            if(strcmp(obj.mode,'useTemperatureAsIndependentFeature') == 1)
                proj = [mu proj];
            end
                
            %% train model
            fprintf('[3/3] Classificator Training\n');
            obj.svm = obj.trainClassifier(proj, labels);
        end
        
        function [model] = learnFeatures(obj, featureName, seqWT)
            model = learnFeatures@BoFW2MetaFunctionality(obj, featureName, seqWT);
        end
        
        function [model] = learnDualFeatures(obj, featureName, featC1, featC2)
            model = learnDualFeatures@BoFW2MetaFunctionality(obj, featureName, featC1, featC2);
        end
        
        function [ projectedData ] = projectData(obj, seqWT)
            projectedData = projectData@BoFW2MetaFunctionality(obj, seqWT, obj.centroids{1});
        end
        
        function [model] = trainClassifier(obj, proj, labels)
            model = trainClassifier@BoFW2MetaFunctionality(obj, proj, labels);
        end         

        function [labels,score, labels2, labels_ch] = applyClassifier(obj, model, proj)
            [labels,score, labels2, labels_ch] = applyClassifier@BoFW2MetaFunctionality(obj, model, proj);
        end
        
        function [ projectedData ] = normalizeDyads(obj, seqWT)
                    [ projectedData ] = normalizeDyads@BoFW2MetaFunctionality(obj, seqWT);
        end
        
        %%
        function [ labels, score, labels2, labels_ch ] = classify( obj, seqWithBGEstimate )
            %% transform and normalize data
            mu = seqWithBGEstimate.mu;
            seqWT = single(FWT_PO_SEQUENCE(seqWithBGEstimate.bgFiltered,obj.L,obj.qmf));
            if(obj.doNormalizeDyads == true)
                %fprintf('--> normalizing dyads\n');
                seqWT = normalizeDyads(obj, seqWT);
            end
            seqWT = seqWT(:,obj.myIndices);    

            
            %% project into frequency bags
            proj = projectData(obj, seqWT);
            clear seqWT;
            if(strcmp(obj.mode,'useTemperatureAsIndependentFeature') == 1)
                proj = [mu proj];
            end
                       
            [labels,score, labels2, labels_ch] = obj.applyClassifier(obj.svm, proj);
            
          
        end
        

        
    end
    
    
end
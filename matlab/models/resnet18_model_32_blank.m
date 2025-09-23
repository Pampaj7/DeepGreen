function lgraph = resnet18_model_32_blank(numClasses)
    % ResNet18 ufficiale MathWorks -> input 32×32 -> testa custom -> reinit totale
    net = resnet18; % usato solo per la topologia
    lgraph = layerGraph(net);
    
    % Input 32×32 (nome layer input ResNet18: 'data')
    inLayer = imageInputLayer([32 32 3], 'Name','data', 'Normalization','none');
    lgraph  = replaceLayer(lgraph, 'data', inLayer);
    
    % Testa: FC + classification
    lgraph = replaceLayer(lgraph, 'fc1000', fullyConnectedLayer(numClasses, 'Name','fc'));
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', classificationLayer('Name','cls'));
    
    % Re-init completa: Conv/FC (He), BN (default)
    lgraph = reinit_all_learnables(lgraph);
end

% =============== utils di re-inizializzazione ===============
function lgraph = reinit_all_learnables(lgraph)
    L = lgraph.Layers;
    for i = 1:numel(L)
        Li = L(i);
        if isa(Li,'nnet.cnn.layer.Convolution2DLayer')
            sz = size(Li.Weights);  fanIn = prod(sz(1:3));
            W = randn(sz,'single') * sqrt(2/fanIn);
            b = zeros(size(Li.Bias),'single');
            L(i) = convolution2dLayer(Li.FilterSize, Li.NumFilters, ...
                'Name',Li.Name, 'Stride',Li.Stride, 'Padding',Li.PaddingSize, ...
                'DilationFactor',Li.DilationFactor, ...
                'BiasLearnRateFactor',Li.BiasLearnRateFactor, ...
                'WeightLearnRateFactor',Li.WeightLearnRateFactor, ...
                'BiasL2Factor',Li.BiasL2Factor, ...
                'WeightL2Factor',Li.WeightL2Factor, ...
                'Weights',W, 'Bias',b);
        elseif isa(Li,'nnet.cnn.layer.FullyConnectedLayer')
            sz = size(Li.Weights);  fanIn = sz(2);
            W = randn(sz,'single') * sqrt(2/fanIn);
            b = zeros(size(Li.Bias),'single');
            L(i) = fullyConnectedLayer(Li.OutputSize, 'Name',Li.Name, ...
                'BiasLearnRateFactor',Li.BiasLearnRateFactor, ...
                'WeightLearnRateFactor',Li.WeightLearnRateFactor, ...
                'BiasL2Factor',Li.BiasL2Factor, ...
                'WeightL2Factor',Li.WeightL2Factor, ...
                'Weights',W, 'Bias',b);
        elseif isa(Li,'nnet.cnn.layer.BatchNormalizationLayer')
            L(i) = batchNormalizationLayer( ...
                'Name',Li.Name, 'Epsilon',Li.Epsilon, ... %'Momentum',Li.Momentum, ... % MATLAB R2023a doesn't expose (old style) Momentum or (new style) MovingAverageMomentum parameter
                'ScaleLearnRateFactor',Li.ScaleLearnRateFactor, ...
                'OffsetLearnRateFactor',Li.OffsetLearnRateFactor, ...
                'ScaleL2Factor',Li.ScaleL2Factor, 'OffsetL2Factor',Li.OffsetL2Factor);
        end
    end
    lgraph = replace_layers_preserve_connections(lgraph, L);
end

function lgraph = replace_layers_preserve_connections(lgraph, newLayers)
    old = lgraph.Layers;
    for i = 1:numel(old)
        lgraph = replaceLayer(lgraph, old(i).Name, newLayers(i));
    end
end
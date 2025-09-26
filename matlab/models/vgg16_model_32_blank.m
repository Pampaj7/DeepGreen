function lgraph = vgg16_model_32_blank(numClasses)
    % Official MathWorks VGG16 > input 32×32 > custom head
    layers = vgg16(Weights='none');
    lgraph = layerGraph(layers);
    
    %               Input 32×32 (input layer name in VGG16: 'input')
    inLayer = imageInputLayer([32 32 3], Name='input', Normalization='none');
    lgraph  = replaceLayer(lgraph, 'input', inLayer);
    
    % Head: flatten + ... + fc + softmax + output
    layersToRemove = {'fc6','relu6','drop6', ...
                      'fc7','relu7','drop7', ...
                      'fc8','prob','output'};
    lgraph = removeLayers(lgraph,layersToRemove);

    newLayers = [
        flattenLayer(Name='flatten')                  % new
        fullyConnectedLayer(4096,Name='fc6')
        reluLayer(Name='relu6')
        dropoutLayer(0.5,Name='drop6')
        fullyConnectedLayer(4096,Name='fc7')
        reluLayer(Name='relu7')
        dropoutLayer(0.5,Name='drop7')
        fullyConnectedLayer(numClasses,Name='fc')     % changed
        softmaxLayer(Name='softmax')                  % new
        classificationLayer(Name='cls')               % new
    ];

    lgraph = addLayers(lgraph,newLayers);
    lgraph = connectLayers(lgraph,'pool5','flatten');
    
    %lgraph = reinit_all_learnables(lgraph);
end

% % =============== utils di re-inizializzazione ===============
% function lgraph = reinit_all_learnables(lgraph)
%     L = lgraph.Layers;
%     for i = 1:numel(L)
%         Li = L(i);
%         if isa(Li,'nnet.cnn.layer.Convolution2DLayer')
%             sz = size(Li.Weights);  fanIn = prod(sz(1:3));
%             W = randn(sz,'single') * sqrt(2/fanIn);
%             b = zeros(size(Li.Bias),'single');
%             L(i) = convolution2dLayer(Li.FilterSize, Li.NumFilters, ...
%                 'Name',Li.Name, 'Stride',Li.Stride, 'Padding',Li.PaddingSize, ...
%                 'DilationFactor',Li.DilationFactor, ...
%                 'BiasLearnRateFactor',Li.BiasLearnRateFactor, ...
%                 'WeightLearnRateFactor',Li.WeightLearnRateFactor, ...
%                 'BiasL2Factor',Li.BiasL2Factor, ...
%                 'WeightL2Factor',Li.WeightL2Factor, ...
%                 'Weights',W, 'Bias',b);
%         elseif isa(Li,'nnet.cnn.layer.FullyConnectedLayer')
%             sz = size(Li.Weights);  fanIn = sz(2);
%             W = randn(sz,'single') * sqrt(2/fanIn);
%             b = zeros(size(Li.Bias),'single');
%             L(i) = fullyConnectedLayer(Li.OutputSize, 'Name',Li.Name, ...
%                 'BiasLearnRateFactor',Li.BiasLearnRateFactor, ...
%                 'WeightLearnRateFactor',Li.WeightLearnRateFactor, ...
%                 'BiasL2Factor',Li.BiasL2Factor, ...
%                 'WeightL2Factor',Li.WeightL2Factor, ...
%                 'Weights',W, 'Bias',b);
%         % VGG16 doesn't use BatchNormalizationLayer
%         % elseif isa(Li,'nnet.cnn.layer.BatchNormalizationLayer')
%         %     L(i) = batchNormalizationLayer( ...
%         %         'Name',Li.Name, 'Epsilon',Li.Epsilon, ... %'Momentum',Li.Momentum, ... % MATLAB R2023a doesn't expose (old style) Momentum or (new style) MovingAverageMomentum parameter
%         %         'ScaleLearnRateFactor',Li.ScaleLearnRateFactor, ...
%         %         'OffsetLearnRateFactor',Li.OffsetLearnRateFactor, ...
%         %         'ScaleL2Factor',Li.ScaleL2Factor, 'OffsetL2Factor',Li.OffsetL2Factor);
%         end
%     end
%     lgraph = replace_layers_preserve_connections(lgraph, L);
% end
% 
% function lgraph = replace_layers_preserve_connections(lgraph, newLayers)
%     old = lgraph.Layers;
%     for i = 1:numel(old)
%         lgraph = replaceLayer(lgraph, old(i).Name, newLayers(i));
%     end
% end
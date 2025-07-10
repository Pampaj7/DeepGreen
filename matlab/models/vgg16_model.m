function lgraph = vgg16_model(numClasses)
    net = vgg16;
    lgraph = layerGraph(net);

    % Sostituisci l’ultimo fully connected layer (fc8)
    lgraph = replaceLayer(lgraph, 'fc8', ...
        fullyConnectedLayer(numClasses, ...
            'Name', 'new_fc', ...
            'WeightLearnRateFactor', 10, ...
            'BiasLearnRateFactor', 10));

    % Sostituisci il softmax e l’output layer
    lgraph = replaceLayer(lgraph, 'prob', softmaxLayer('Name', 'new_softmax'));
    lgraph = replaceLayer(lgraph, 'output', classificationLayer('Name', 'new_output'));
end
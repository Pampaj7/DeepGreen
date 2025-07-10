function lgraph = resnet18_model(numClasses)
    net = resnet18;
    lgraph = layerGraph(net);

    % Replace FC layer
    lgraph = replaceLayer(lgraph, 'fc1000', ...
        fullyConnectedLayer(numClasses, 'Name','fc1000'));

    % Replace classification layer
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', ...
        classificationLayer('Name','new_output'));
end
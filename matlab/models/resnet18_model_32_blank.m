function lgraph = resnet18_model_32_blank(numClasses)
    % Official MathWorks ResNet18 > input 32×32 > custom head
    lgraph = resnet18(Weights='none');
    
    % Input 32×32 (input layer name in ResNet18: 'data')
    inLayer = imageInputLayer([32 32 3], Name='data', Normalization='none');
    lgraph  = replaceLayer(lgraph, 'data', inLayer);
    
    % Head: FC + classification
    lgraph = replaceLayer(lgraph, 'fc1000', ...
        fullyConnectedLayer(numClasses, Name='fc'));
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', ...
        classificationLayer(Name='cls'));
end
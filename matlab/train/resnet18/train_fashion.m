thisFileDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisFileDir, '..', '..')));


function train_resnet18_fashion(datasetDir, outMat, epochs, batchSize)
if nargin<1||isempty(datasetDir), datasetDir = 'data/fashion_mnist_png'; end
if nargin<2||isempty(outMat),     outMat     = 'checkpoints/resnet18_fashion_matlab.mat'; end
if nargin<3||isempty(epochs),     epochs     = 30; end
if nargin<4||isempty(batchSize),  batchSize  = 128; end


trainDir = fullfile(datasetDir,'train');
valDir   = fullfile(datasetDir,'test');
assert(isfolder(trainDir)&&isfolder(valDir));

imdsTrain = imageDatastore(trainDir,'IncludeSubfolders',true,'LabelSource','foldernames');
imdsVal   = imageDatastore(valDir,  'IncludeSubfolders',true,'LabelSource','foldernames');
numClasses = numel(categories(imdsTrain.Labels));

augTrain = augmentedImageDatastore([32 32], imdsTrain, 'ColorPreprocessing','gray2rgb');
augVal   = augmentedImageDatastore([32 32], imdsVal,   'ColorPreprocessing','gray2rgb');

lgraph = resnet18_model_32_blank(numClasses);

opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, 'MiniBatchSize',batchSize, 'MaxEpochs',epochs, ...
    'Shuffle','every-epoch', 'ValidationData',augVal, ...
    'ValidationFrequency',max(1,floor(numel(imdsTrain.Files)/batchSize)), ...
    'Verbose',true, 'Plots','none');

net = trainNetwork(augTrain, lgraph, opts);
if ~isfolder(fileparts(outMat)), mkdir(fileparts(outMat)); end
save(outMat,'net','-v7.3');
end
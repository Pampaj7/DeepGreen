function train_cifar100(datasetDir, outMat, epochs, batchSize)
% ResNet18 — CIFAR100 — input 32×32 — pesi azzerati (He), BN reset
% Esempio:
% matlab -batch "addpath(genpath('matlab')); train_cifar100('data/cifar100_png','checkpoints/resnet18_cifar100_matlab.mat',30,128); exit"

% --------- default args ---------
if nargin<1||isempty(datasetDir), datasetDir = 'data/cifar100_png'; end
if nargin<2||isempty(outMat),     outMat     = 'checkpoints/resnet18_cifar100_matlab.mat'; end
if nargin<3||isempty(epochs),     epochs     = 30; end
if nargin<4||isempty(batchSize),  batchSize  = 128; end

% --------- path robusto (models/, utils/, ecc.) ---------
thisFileDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisFileDir, '..', '..')));  % -> matlab/

% --------- DATA ---------
trainDir = fullfile(datasetDir,'train');
valDir   = fullfile(datasetDir,'test');
if ~isfolder(valDir), valDir = fullfile(datasetDir,'val'); end
assert(isfolder(trainDir)&&isfolder(valDir), 'Missing train/test folders in %s', datasetDir);

imdsTrain = imageDatastore(trainDir,'IncludeSubfolders',true,'LabelSource','foldernames');
imdsVal   = imageDatastore(valDir,  'IncludeSubfolders',true,'LabelSource','foldernames');
numClasses = numel(categories(imdsTrain.Labels));
fprintf('Found %d classes in training set.\n', numClasses);

augTrain = augmentedImageDatastore([32 32], imdsTrain);
augVal   = augmentedImageDatastore([32 32], imdsVal);

% --------- MODEL ---------
% definito in matlab/models/model32.m
lgraph = resnet18_model_32_blank(numClasses);

% --------- TRAIN ---------
opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, 'MiniBatchSize',batchSize, 'MaxEpochs',epochs, ...
    'Shuffle','every-epoch', 'ValidationData',augVal, ...
    'ValidationFrequency',max(1,floor(numel(imdsTrain.Files)/batchSize)), ...
    'Verbose',true, 'Plots','none');

fprintf('Starting training ResNet18 (32x32) …\n');
net = trainNetwork(augTrain, lgraph, opts);

% --------- SAVE ---------
if ~isfolder(fileparts(outMat)), mkdir(fileparts(outMat)); end
save(outMat,'net','-v7.3');
fprintf('Saved checkpoint to %s\n', outMat);
end
% VGG16 — CIFAR100 — input 32×32 — pesi azzerati (He), BN reset
%
% Instructions to run:
% 1) Locate yourself into main project folder (e.g. DeepGreen, i.e. where
% .git is located)
% 2) Before run, set the matlab folder (and subdirectories) to matlab's paths via:
% >> addpath(genpath('matlab'));
% 3) Run the function via (don't change location)
% >> vgg16.train_cifar100('data/cifar100_png','matlab/checkpoints/vgg16_cifar100_matlab.mat',30,128);
% 
% Alternatvely, run :
% $ matlab -batch "; vgg16.train_cifar100('data/cifar100_png','matlab/checkpoints/vgg16_cifar100_matlab.mat',30,128); exit"
%
function train_cifar100(datasetDir, outMat, epochs, batchSize)
    % --------- default args ---------
    if nargin<1||isempty(datasetDir), datasetDir = 'data/cifar100_png'; end
    if nargin<2||isempty(outMat),     outMat     = 'matlab/checkpoints/vgg16_cifar100_matlab.mat'; end
    if nargin<3||isempty(epochs),     epochs     = 30; end
    if nargin<4||isempty(batchSize),  batchSize  = 128; end
    emissionOutputDir = 'matlab/emissions';
    emissionFileName = 'vgg16_cifar100';
    
    % --------- DATA ---------
    trainDir = fullfile(datasetDir,'train');
    valDir   = fullfile(datasetDir,'test');
    assert(isfolder(trainDir)&&isfolder(valDir), 'Missing train/test folders in %s', datasetDir);
    
    imdsTrain = imageDatastore(trainDir,'IncludeSubfolders',true,'LabelSource','foldernames');
    imdsVal   = imageDatastore(valDir,  'IncludeSubfolders',true,'LabelSource','foldernames');
    numClasses = numel(categories(imdsTrain.Labels));
    fprintf('Found %d classes in training set.\n', numClasses);

    augTrain = augmentedImageDatastore([32 32], imdsTrain);
    augVal   = augmentedImageDatastore([32 32], imdsVal);
    
    % --------- MODEL ---------
    % definito in matlab/models/model32.m
    lgraph = vgg16_model_32_blank(numClasses);
    
    % --------- TRAIN CONF ---------
    opts = trainingOptions('adam', ...
        'InitialLearnRate',1e-4, 'MiniBatchSize',batchSize, 'MaxEpochs',epochs, ...
        'Shuffle','every-epoch', 'ValidationData',augVal, ...
        'ValidationFrequency',max(1,floor(numel(imdsTrain.Files)/batchSize)), ...
        'Verbose',true, 'Plots','none');
    
    % --------- REMOVE EXISTING EMISSION FILES ---------
    trainEmissionFile = strcat(emissionFileName, '_train.csv');
    if isfile(fullfile(emissionOutputDir, trainEmissionFile))
        delete(fullfile(emissionOutputDir, trainEmissionFile));
    end
    
    % --------- TRAIN LOOP ---------
    fprintf('Starting training VGG16 on CIFAR-100 (32x32) …\n');
    py.tracker_control.Tracker.start_tracker(emissionOutputDir, trainEmissionFile);
    net = trainNetwork(augTrain, lgraph, opts);
    py.tracker_control.Tracker.stop_tracker();

    % --------- SAVE MODEL ---------
    if ~isfolder(fileparts(outMat)), mkdir(fileparts(outMat)); end
    save(outMat,'net','-v7.3');
    fprintf('Saved checkpoint to %s\n', outMat);
end
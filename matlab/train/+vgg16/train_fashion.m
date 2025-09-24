% VGG16 — FashionMNIST — input 32×32 — pesi azzerati (He), BN reset
%
% Instructions to run:
% 1) Locate yourself into main project folder (e.g. DeepGreen, i.e. where
% .git is located)
% 2) Before run, set the matlab folder (and subdirectories) to matlab's paths via:
% $ addpath(genpath('matlab'));
% 3) Run the function via (don't change location)
% $ vgg16.train_fashion('data/fashion_mnist_png','matlab/checkpoints/vgg16_fashion_matlab.mat',30,128);
% 
% Alternatvely, run :
% $ matlab -batch "; vgg16.train_fashion('data/fashion_mnist_png','matlab/checkpoints/vgg16_fashion_matlab.mat',30,128); exit"
%
function train_fashion(datasetDir, outMat, epochs, batchSize)
    % --------- default args ---------
    if nargin<1||isempty(datasetDir), datasetDir = 'data/fashion_mnist_png'; end
    if nargin<2||isempty(outMat),     outMat     = 'matlab/checkpoints/vgg16_fashion_matlab.mat'; end
    if nargin<3||isempty(epochs),     epochs     = 30; end
    if nargin<4||isempty(batchSize),  batchSize  = 128; end
    
    % --------- DATA ---------
    trainDir = fullfile(datasetDir,'train');
    valDir   = fullfile(datasetDir,'test');
    assert(isfolder(trainDir)&&isfolder(valDir), 'Missing train/test folders in %s', datasetDir);
    
    imdsTrain = imageDatastore(trainDir,'IncludeSubfolders',true,'LabelSource','foldernames');
    imdsVal   = imageDatastore(valDir,  'IncludeSubfolders',true,'LabelSource','foldernames');
    numClasses = numel(categories(imdsTrain.Labels));
    fprintf('Found %d classes in training set.\n', numClasses);

    augTrain = augmentedImageDatastore([32 32], imdsTrain, 'ColorPreprocessing','gray2rgb');
    augVal   = augmentedImageDatastore([32 32], imdsVal,   'ColorPreprocessing','gray2rgb');
    
    % --------- MODEL ---------
    % definito in matlab/models/model32.m
    lgraph = vgg16_model_32_blank(numClasses);
    
    % --------- TRAIN ---------
    opts = trainingOptions('adam', ...
        'InitialLearnRate',1e-4, 'MiniBatchSize',batchSize, 'MaxEpochs',epochs, ...
        'Shuffle','every-epoch', 'ValidationData',augVal, ...
        'ValidationFrequency',max(1,floor(numel(imdsTrain.Files)/batchSize)), ...
        'Verbose',true, 'Plots','none');
    
    fprintf('Starting training VGG16 on FashionMNIST (32x32) …\n');
    py.tracker_control.Tracker.start_tracker('matlab/emissions','vgg16_fashion.csv');
    net = trainNetwork(augTrain, lgraph, opts);
    py.tracker_control.Tracker.stop_tracker();

    % --------- SAVE ---------
    if ~isfolder(fileparts(outMat)), mkdir(fileparts(outMat)); end
    save(outMat,'net','-v7.3');
    fprintf('Saved checkpoint to %s\n', outMat);
end
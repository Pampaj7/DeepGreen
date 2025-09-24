% VGG16 — Tiny ImageNet — input 32×32 — pesi azzerati (He), BN reset
%
% Instructions to run:
% 1) Locate yourself into main project folder (e.g. DeepGreen, i.e. where
% .git is located)
% 2) Before run, set the matlab folder (and subdirectories) to matlab's paths via:
% >> addpath(genpath('matlab'));
% 3) Run the function via (don't change location)
% >> vgg16.train_tiny('data/tiny_imagenet_png','vgg16_tiny','matlab/checkpoints/vgg16_tiny_matlab.mat',30,128);
% 
% Alternatvely, run :
% $ matlab -batch "; vgg16.train_tiny('data/tiny_imagenet_png','vgg16_tiny','matlab/checkpoints/vgg16_tiny_matlab.mat',30,128); exit"
%
function train_tiny(datasetDir, emissionFileName, outMat, epochs, batchSize)
    % --------- default args ---------
    if nargin<1||isempty(datasetDir),       datasetDir          = 'data/tiny_imagenet_png'; end
    if nargin<2||isempty(emissionFileName), emissionFileName    = 'vgg16_tiny'; end
    if nargin<3||isempty(outMat),           outMat              = 'matlab/checkpoints/vgg16_tiny_matlab.mat'; end
    if nargin<4||isempty(epochs),           epochs              = 30; end
    if nargin<5||isempty(batchSize),        batchSize           = 128; end
    emissionOutputDir = 'matlab/emissions';
    
    % --------- DATA ---------
    trainDir = fullfile(datasetDir,'train');
    testDir  = fullfile(datasetDir,'val');
    assert(isfolder(trainDir)&&isfolder(testDir), 'Missing train/test folders in %s', datasetDir);
    
    imdsTrain = imageDatastore(trainDir,'IncludeSubfolders',true,'LabelSource','foldernames');
    imdsTest  = imageDatastore(testDir, 'IncludeSubfolders',true,'LabelSource','foldernames');
    numClasses = numel(categories(imdsTrain.Labels));
    fprintf('Found %d classes in training set.\n', numClasses);

    % Some images of Tiny ImageNet are grayscale: convertion to RGB is needed
    augTrain = augmentedImageDatastore([32 32], imdsTrain,'ColorPreprocessing','gray2rgb');
    augTest  = augmentedImageDatastore([32 32], imdsTest, 'ColorPreprocessing','gray2rgb');

    % --------- LOSS ---------
    % With trainNetwork function the used loss depends by the last layer:
    % - if classificationLayer  then cross-entropy is used
    % - if regressionLayer      then Mean Squared Error is used
    
    % --------- MODEL ---------
    % definito in matlab/models/model32.m
    lgraph = vgg16_model_32_blank(numClasses);
    
    % --------- TRAIN CONF ---------
    opts = trainingOptions('adam', ...
        'InitialLearnRate',1e-4, ...
        'MiniBatchSize',batchSize, ...
        'MaxEpochs',epochs, ...
        'Shuffle','every-epoch', ...
        'Verbose',true, ...
        'Plots','none');
    
    % --------- REMOVE EXISTING EMISSION FILES ---------
    trainEmissionFile = strcat(emissionFileName, '_train.csv');
    if isfile(fullfile(emissionOutputDir, trainEmissionFile))
        delete(fullfile(emissionOutputDir, trainEmissionFile));
    end
    
    % --------- TRAIN LOOP ---------
    fprintf('Starting training VGG16 on Tiny ImageNet (32x32) …\n');
    py.tracker_control.Tracker.start_tracker(emissionOutputDir, trainEmissionFile);
    net = trainNetwork(augTrain, lgraph, opts);
    py.tracker_control.Tracker.stop_tracker();

    % Predizione sul test set
    YPred = classify(net, augTest);
    YTest = imdsTest.Labels;
    accuracy = mean(YPred == YTest) * 100;
    disp("Test accuracy: " + accuracy + "%");

    % --------- SAVE MODEL ---------
    if ~isfolder(fileparts(outMat)), mkdir(fileparts(outMat)); end
    save(outMat,'net','-v7.3');
    fprintf('Saved checkpoint to %s\n', outMat);
end
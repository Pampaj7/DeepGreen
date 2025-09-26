% VGG16 — Tiny ImageNet — input 32×32 — pesi azzerati (He), BN reset
%
% Instructions to run at "settings_for_matlab.txt".
%
function train_tiny(datasetDir, emissionFileName, outMat, img_size, epochs, batchSize, lr)
    % --------- default args ---------
    if nargin<1||isempty(datasetDir),       datasetDir          = 'data/tiny_imagenet_png'; end
    if nargin<2||isempty(emissionFileName), emissionFileName    = 'vgg16_tiny'; end
    if nargin<3||isempty(outMat),           outMat              = 'matlab/checkpoints/vgg16_tiny_matlab.mat'; end
    if nargin<4||isempty(img_size),         img_size            = [32 32]; end
    if nargin<5||isempty(epochs),           epochs              = 30; end
    if nargin<6||isempty(batchSize),        batchSize           = 128; end
    if nargin<7||isempty(lr),               lr                  = 1e-4; end
    
    % --------- DATA ---------
    trainDir = fullfile(datasetDir,'train');
    testDir  = fullfile(datasetDir,'val');
    assert(isfolder(trainDir)&&isfolder(testDir), 'Missing train/test folders in %s', datasetDir);
    
    imdsTrain = imageDatastore(trainDir,IncludeSubfolders=true,LabelSource='foldernames');
    imdsTest  = imageDatastore(testDir, IncludeSubfolders=true,LabelSource='foldernames');
    numClasses = numel(categories(imdsTrain.Labels));
    fprintf('Found %d classes in training set.\n', numClasses);

    % Resize and convert to RGB (if necessary)
    % Some images of Tiny ImageNet are grayscale: convertion to RGB is needed
    augTrain = augmentedImageDatastore(img_size, imdsTrain,ColorPreprocessing='gray2rgb');
    augTest  = augmentedImageDatastore(img_size, imdsTest, ColorPreprocessing='gray2rgb');

    % Normalize from [0-255] to [0-1]
    normalizeFcn = @(data) setfield(data,'input', ...
        cellfun(@(img) single(img)./255, data.input, UniformOutput=false) );
    augTrain = transform(augTrain,normalizeFcn);
    augTest  = transform(augTest, normalizeFcn);

    % --------- LOSS ---------
    % With trainNetwork function the used loss depends by the last layer:
    % - if classificationLayer  then cross-entropy is used
    % - if regressionLayer      then Mean Squared Error is used
    
    % --------- MODEL ---------
    % definito in matlab/models/model32.m
    lgraph = vgg16_model_32_blank(numClasses);
    
    % --------- TRAIN CONF ---------
    opts = trainingOptions('adam', ...
        InitialLearnRate=lr, ...
        MiniBatchSize=batchSize, ...
        MaxEpochs=1, ...
        Shuffle='every-epoch', ... % shuffle is applied BEFORE every epoch
        Verbose=true, ...
        Plots='none');

    % --------- TRACKER ---------
    tracker_control = py.importlib.import_module('tracker_control');
    emissionOutputDir = 'matlab/emissions';
    
    % --------- REMOVE EXISTING EMISSION FILES ---------
    trainEmissionFile = strcat(emissionFileName, '_train.csv');
    if isfile(fullfile(emissionOutputDir, trainEmissionFile))
        delete(fullfile(emissionOutputDir, trainEmissionFile));
    end
    testEmissionFile = strcat(emissionFileName, '_test.csv');
    if isfile(fullfile(emissionOutputDir, testEmissionFile))
        delete(fullfile(emissionOutputDir, testEmissionFile));
    end
    
    % --------- TRAIN LOOP ---------
    fprintf('Starting training VGG16 on Tiny ImageNet (32x32) …\n');
    for k = 1:epochs
        fprintf('--- Epoch %d/%d ---\n',k,epochs)

        % Training
        tracker_control.Tracker.start_tracker(emissionOutputDir, trainEmissionFile);
        net = trainNetwork(augTrain, lgraph, opts);
        tracker_control.Tracker.stop_tracker();

        % Testing
        tracker_control.Tracker.start_tracker(emissionOutputDir, testEmissionFile);
        YPred = classify(net, augTest, ...
            MiniBatchSize=batchSize, ...
            ExecutionEnvironment='gpu');
        tracker_control.Tracker.stop_tracker();
        
        % Print details
        accuracy = mean(YPred == imdsTest.Labels) * 100;
        disp("Test accuracy: " + accuracy + "%");

        % Convert net (DAGNetwork) to layer graph 
        % Necessary for next trainNetwork epoch
        lgraph = layerGraph(net);
    end

    % --------- SAVE MODEL ---------
    if ~isfolder(fileparts(outMat)), mkdir(fileparts(outMat)); end
    save(outMat,'net','-v7.3');
    fprintf('Saved checkpoint to %s\n', outMat);
end
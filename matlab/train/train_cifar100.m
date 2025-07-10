addpath(fullfile(pwd, '..', 'utils'));
addpath(fullfile(pwd, '..', 'models'));

data_path = fullfile('/Users/pampaj/PycharmProjects/DeepGreen', 'data', 'cifar100_png');
train_path = fullfile(data_path, 'train');
test_path  = fullfile(data_path, 'test');

train_ds = load_data(train_path);
test_ds  = load_data(test_path);
numClasses = numel(categories(train_ds.Labels));

% Resize images to match ResNet18 input size (224x224x3)
inputSize = [224 224 3];
train_aug = augmentedImageDatastore(inputSize, train_ds);
test_aug  = augmentedImageDatastore(inputSize, test_ds);

% Choose model
%lgraph = resnet18_model(numClasses);
lgraph = vgg16_model(numClasses); % ← switch here if needed

options = trainingOptions("adam", ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',64, ...
    'ValidationData',test_aug, ...
    'Verbose',false, ...
    'Plots','training-progress');

trained_net = trainNetwork(train_aug, lgraph, options);
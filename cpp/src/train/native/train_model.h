#ifndef TRAIN_MODEL_H
#define TRAIN_MODEL_H
#include <iostream>
#include <torch/torch.h>

#include "dataset/ImageFolder.h"
#include "cnn_function.h"
#include "cnn_setup.h"
#include "dataset_transforms.h"
#include "utils.h"
#include "python/PythonTracker.h"


template <typename Model, typename Dataset>
void train_model(const std::string& outputFileName, const char* dataRootRelativePath, const char* classesJson,
    Model& model, int32_t imgResize,
    const int32_t trainBatchSize, const int32_t testBatchSize, const int32_t numberOfEpochs)
{
    // device (CPU or GPU)
    torch::Device device = CNNSetup::get_device_available();


    // transformations
    auto transform_list = std::vector<TorchTrasformPtr>
    {
        //std::make_shared<torch::data::transforms::Normalize<>>(Dataset::getMean(), Dataset::getStd()),
        std::make_shared<DatasetTransforms::ResizeTo>(imgResize, imgResize)
    };
    if (Dataset::isGrayscale())
        transform_list.push_back(std::make_shared<DatasetTransforms::ReplicateChannels>());

    auto composedTransform = DatasetTransforms::Compose(transform_list);


    // dataset
    std::string kDataRootFullPath = Utils::join_paths(PROJECT_SOURCE_DIR, dataRootRelativePath);
    std::string kClassesFullPath = Utils::join_paths(kDataRootFullPath, classesJson);

    std::cout << "Preparing " << Dataset::getDatasetName() << " for training...";
    ImageFolder<Dataset> train_set{kDataRootFullPath, kClassesFullPath, true};
    auto train_set_transformed =
        train_set
            .map(composedTransform)
            .map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_set_transformed.size().value();
    std::cout << " Done." << std::endl;

    std::cout << "Preparing " << Dataset::getDatasetName() << " for testing...";
    ImageFolder<Dataset> test_set{kDataRootFullPath, kClassesFullPath, false};
    auto test_set_transformed =
        test_set
            .map(composedTransform)
            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_set_transformed.size().value();
    std::cout << " Done." << std::endl;


    // dataloader
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>( // same as torch.utils.data.DataLoader.shuffle(true)
            std::move(train_set_transformed),
            torch::data::DataLoaderOptions()
                    .batch_size(trainBatchSize)
                    .workers(2)
                    .enforce_ordering(true)); // same as torch.utils.data.DataLoader.in_order(true)
    auto test_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>( // same as torch.utils.data.DataLoader.shuffle(false)
            std::move(test_set_transformed),
            torch::data::DataLoaderOptions()
                    .batch_size(testBatchSize)
                    .workers(2)
                    .enforce_ordering(true)); // same as torch.utils.data.DataLoader.in_order(true)


    // loss
    torch::nn::CrossEntropyLoss criterion{};

    // model
    model->to(device);

    // optimizer
    torch::optim::Adam optimizer(
        model->parameters(/* recurse = */ true),
        torch::optim::AdamOptions(1e-4)
    );


    // Remove existing emission file
    const std::string outputDir = Utils::join_paths(PROJECT_SOURCE_DIR, "emissions");
    const std::string trainOutputFile = outputFileName + "_train.csv";
    Utils::removeFileIfExists(Utils::join_paths(outputDir, trainOutputFile));
    const std::string testOutputFile = outputFileName + "_test.csv";
    Utils::removeFileIfExists(Utils::join_paths(outputDir, testOutputFile));

    // tracker
    PythonTracker::initializeTracker();

    // training loop
    for (uint32_t epoch = 1; epoch <= numberOfEpochs; ++epoch) {
        std::printf("Epoch {%u}/{%d}\n",
            epoch,
            numberOfEpochs);

        PythonTracker::startTracker(outputDir, trainOutputFile);
        float train_loss = CNNFunction::train(model, device, *train_loader, optimizer, train_dataset_size, criterion);
        PythonTracker::stopTracker();

        PythonTracker::startTracker(outputDir, testOutputFile);
        auto test_loss_and_acc = CNNFunction::test(model, device, *test_loader, test_dataset_size, criterion);
        PythonTracker::stopTracker();

        std::printf(
            "Train Loss: %.4f | Test Loss:  %.4f | Accuracy: %.2f%%\n",
            train_loss,
            test_loss_and_acc.at(0),
            test_loss_and_acc.at(1));
    }

    PythonTracker::finalizeTracker();
}



#endif //TRAIN_MODEL_H

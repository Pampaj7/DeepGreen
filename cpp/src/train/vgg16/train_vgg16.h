#ifndef TRAIN_VGG16_H
#define TRAIN_VGG16_H
#include <iostream>
#include <torch/torch.h>

#include "dataset/ImageFolder.h"
#include "cnn_function.h"
#include "cnn_setup.h"
#include "dataset_transforms.h"
#include "utils.h"

constexpr int32_t kVggMinImageSize = 32;

template <typename Dataset>
void train_vgg16(const char* kDataRootRelativePath, const char* kClassesJson, const char* model_dataset_filename,
    const int64_t kTrainBatchSize, int64_t kTestBatchSize, int64_t kNumberOfEpochs)
{
    // device (CPU or GPU)
    torch::Device device = CNNSetup::get_device_available();

    // dataset
    std::string kDataRootFullPath = Utils::join_paths(PROJECT_SOURCE_DIR, kDataRootRelativePath);
    std::string kClassesFullPath = Utils::join_paths(kDataRootFullPath, kClassesJson);

    // transformations
    auto transform_list = std::vector<TorchTrasformPtr> // TODO: queue
    {
        std::make_shared<torch::data::transforms::Normalize<>>(Dataset::getMean(), Dataset::getStd())
    };
    for (auto transform : transform_list)
        std::cout << transform << std::endl;
    // Vgg-16 richiede immagini a dimensione almeno 32x32, altrimenti l'output di feature è a dimensione nulla
    if (Dataset::getImageHeight() < kVggMinImageSize || Dataset::getImageWidth() < kVggMinImageSize)
        transform_list.push_back(
            std::make_shared<DatasetTransforms::ResizeTo>(
                Dataset::getImageHeight() < kVggMinImageSize ? kVggMinImageSize : Dataset::getImageHeight(), // ridimensiono solo il necessario
                Dataset::getImageWidth() < kVggMinImageSize ? kVggMinImageSize : Dataset::getImageWidth() // ridimensiono solo il necessario
            )
        );
    for (auto transform : transform_list)
        std::cout << transform << std::endl;
    if (Dataset::isGrayscale())
        transform_list.push_back(std::make_shared<DatasetTransforms::ReplicateChannels>());
    for (auto transform : transform_list)
        std::cout << transform << std::endl;

    auto composedTransform = DatasetTransforms::Compose(transform_list);

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
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_set_transformed),
            torch::data::DataLoaderOptions()
            .batch_size(kTrainBatchSize)
            .workers(2)
            .enforce_ordering(true));   // TODO: is same of shuffle?
    auto test_loader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(test_set_transformed),
            torch::data::DataLoaderOptions()
            .batch_size(kTestBatchSize)
            .workers(2)
            .enforce_ordering(false)); // TODO: is same of shuffle?


    // model
    torch::jit::script::Module model = CNNSetup::load_model(
        Utils::join_paths(CMAKE_BINARY_DIR, model_dataset_filename));
    model.to(device);


    // loss
    // auto criterion = TODO: vedere se si può modificare options dopo aver creato la loss


    // optimizer
    auto params_list = model.parameters();
    std::vector<torch::Tensor> parameters;
    for (const auto& p : params_list) {
        parameters.push_back(p);
    }
    torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(1e-4));


    // tracker
    // TODO: here create tracker


    // training loop
    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        std::printf("Epoch {%llu}/{%lld}\n",
            epoch,
            kNumberOfEpochs);

        // TODO: tracker.start()
        CNNFunction::train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        CNNFunction::test(model, device, *test_loader, test_dataset_size);
        //TODO: tracker.stop()
    }
}



#endif //TRAIN_VGG16_H
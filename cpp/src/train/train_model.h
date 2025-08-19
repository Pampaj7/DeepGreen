#ifndef TRAIN_MODEL_H
#define TRAIN_MODEL_H
#include <iostream>
#include <torch/torch.h>

#include "dataset/ImageFolder.h"
#include "cnn_function.h"
#include "cnn_setup.h"
#include "dataset_transforms.h"
#include "utils.h"


template <typename Dataset>
void train_model(const char* dataRootRelativePath, const char* classesJson,
    const char* model_dataset_filename, int32_t modelMinImageSize,
    const int32_t trainBatchSize, const int32_t testBatchSize, const int32_t numberOfEpochs)
{
    // device (CPU or GPU)
    torch::Device device = CNNSetup::get_device_available();


    // transformations
    auto transform_list = std::vector<TorchTrasformPtr>
    {
        std::make_shared<torch::data::transforms::Normalize<>>(Dataset::getMean(), Dataset::getStd())
    };
    if (Dataset::getImageHeight() < modelMinImageSize || Dataset::getImageWidth() < modelMinImageSize) // resize only if necessary
        transform_list.push_back(
            std::make_shared<DatasetTransforms::ResizeTo>(
                Dataset::getImageHeight() < modelMinImageSize ? modelMinImageSize : Dataset::getImageHeight(),
                Dataset::getImageWidth() < modelMinImageSize ? modelMinImageSize : Dataset::getImageWidth()
            )
        );
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


    // model
    torch::jit::script::Module model = CNNSetup::load_model(
        Utils::join_paths(CMAKE_BINARY_DIR, model_dataset_filename));
    model.to(device);


    // loss
    torch::nn::CrossEntropyLoss criterion{};


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
    for (uint32_t epoch = 1; epoch <= numberOfEpochs; ++epoch) {
        std::printf("Epoch {%u}/{%d}\n",
            epoch,
            numberOfEpochs);

        // TODO: tracker.start()
        CNNFunction::train(epoch, model, device, *train_loader, optimizer, train_dataset_size, criterion);
        //TODO: tracker.stop()

        // TODO: tracker.start()
        CNNFunction::test(model, device, *test_loader, test_dataset_size, criterion);
        //TODO: tracker.stop()
    }
}



#endif //TRAIN_MODEL_H

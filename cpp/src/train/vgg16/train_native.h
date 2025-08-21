#ifndef TRAIN_NATIVE_H
#define TRAIN_NATIVE_H

#include <iostream>
#include <torch/torch.h>

#include "dataset/ImageFolder.h"
#include "cnn_function.h"
#include "cnn_setup.h"
#include "dataset_transforms.h"
#include "utils.h"
#include "model/VGG16_native.h"

void print_num_parameters(torch::nn::Module& model) {
    std::size_t total_params = 0;
    for (const auto& param : model.parameters(/*recurse=*/true)) {
        total_params += param.numel();
    }
    std::cout << "Numero totale di parametri: " << total_params << std::endl;
}

void print_trainable_parameters(torch::nn::Module& model) {
    std::size_t trainable_params = 0;
    for (const auto& param : model.parameters(/*recurse=*/true)) {
        if (param.requires_grad()) {
            trainable_params += param.numel();
        }
    }
    std::cout << "Numero di parametri addestrabili: "
              << trainable_params << std::endl;
}


template <typename Dataset>
void train_native(const char* dataRootRelativePath, const char* classesJson,
    int32_t modelMinImageSize,
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



    std::cout << "LIBTORCH MODEL" << std::endl;
    // model
    vision::models::VGG16 native_model(100, false);
    native_model->apply(
        [](torch::nn::Module& m) {
            if (auto* conv = m.as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(conv->weight);
            torch::nn::init::zeros_(conv->bias);
            }
            else if (auto* linear = m.as<torch::nn::Linear>()) {
                torch::nn::init::kaiming_normal_(linear->weight);
                torch::nn::init::zeros_(linear->bias);
            }
        }
    );
    std::cout << *native_model << std::endl;
    print_num_parameters(*native_model);
    print_trainable_parameters(*native_model);
    native_model->to(device);

    // loss
    torch::nn::CrossEntropyLoss criterion{};

    // optimizer
    torch::optim::Adam native_optimizer(native_model->parameters(/* recurse = */ true), torch::optim::AdamOptions(1e-4));



    // training loop
    for (uint32_t epoch = 1; epoch <= numberOfEpochs; ++epoch) {
        std::printf("Epoch {%u}/{%d}\n",
            epoch,
            numberOfEpochs);

        CNNFunction::train(epoch, native_model, device, *train_loader, native_optimizer, train_dataset_size, criterion);
        CNNFunction::test(native_model, device, *test_loader, test_dataset_size, criterion);
    }
}


#endif //TRAIN_NATIVE_H

#include <iostream>
#include <torch/torch.h>

#include "dataset/ImageFolder.h"
#include "dataset/FashionMNIST.h"
#include "cnn_function.h"
#include "cnn_setup.h"
#include "dataset_transforms.h"
#include "utils.h"


// Where to find the Fashion-MNIST dataset.
const char* kDataRootRelativePath = "../data/fashion_mnist_png";
const char* kClassesJson = "classes.json";

// The batch size for training.
constexpr int64_t kTrainBatchSize = 64;
// The batch size for testing.
constexpr int64_t kTestBatchSize = 1000; //TODO
// The number of epochs to train.
constexpr int64_t kNumberOfEpochs = 1;



int main() {
    try {
        // device (CPU or GPU)
        torch::Device device = CNNSetup::get_device_available();

        // dataset
        std::string kDataRootFullPath = Utils::join_paths(PROJECT_SOURCE_DIR, kDataRootRelativePath);
        std::string kClassesFullPath = Utils::join_paths(kDataRootFullPath, kClassesJson);

        // transformations
        auto transform_list = std::vector<TorchTrasformPtr>
        {
            std::make_shared<torch::data::transforms::Normalize<>>(FashionMNIST::getMean(), FashionMNIST::getStd()),
            std::make_shared<DatasetTransforms::ResizeTo>(32, 32),  // resize a 32x32 TODO:necessario per i livelli di vgg che altrimenti ottiene output di feature a dimensione nulla
            std::make_shared<DatasetTransforms::ReplicateChannels>()
        };
        auto composedTransformation = DatasetTransforms::Compose(transform_list);

        std::cout << "Preparing Fashion-MNIST for training...";
        ImageFolder<FashionMNIST> train_set{kDataRootFullPath, kClassesFullPath, true};
        auto train_set_transformed =
            train_set
                .map(composedTransformation)
                .map(torch::data::transforms::Stack<>());
        const size_t train_dataset_size = train_set_transformed.size().value();
        std::cout << " Done." << std::endl;

        std::cout << "Preparing Fashion-MNIST for testing...";
        ImageFolder<FashionMNIST> test_set{kDataRootFullPath, kClassesFullPath, false};
        auto test_set_transformed =
            test_set
                .map(composedTransformation)
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
            Utils::join_paths(CMAKE_BINARY_DIR, VGG16_FASHIONMNIST_FILENAME));
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
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
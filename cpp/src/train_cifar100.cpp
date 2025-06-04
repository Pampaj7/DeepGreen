#include <iostream>
#include <torch/torch.h>

#include "CIFAR100.h"
#include "cnn_function.h"
#include "cnn_setup.h"


// Where to find the CIFAR10 dataset.
const char* kDataRoot = "/../data/cifar100_png";
const char* kClassesPath = "/../data/cifar100_png/classes.json";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000; //TODO

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;



int main() {
    try {
        // device (CPU or GPU)
        torch::Device device = CNNSetup::get_device_available();

        // dataset
        CIFAR100 train_set{kDataRoot, kClassesPath, true};
        auto train_set_transformed =
            train_set
                .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                        {0.2470, 0.2434, 0.2616}))
                .map(torch::data::transforms::Stack<>());
        const size_t train_dataset_size = train_set_transformed.size().value();

        CIFAR100 test_set{kDataRoot, kClassesPath, false};
        auto test_set_transformed =
            test_set
                .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                        {0.2470, 0.2434, 0.2616}))
                .map(torch::data::transforms::Stack<>());
        const size_t test_dataset_size = test_set_transformed.size().value();



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
        std::stringstream modelPath;
        modelPath << CMAKE_BINARY_DIR << "/resnet18.pt"; //TODO: necessario binary dir?
        torch::jit::script::Module model = CNNSetup::load_model(modelPath.str()); // TODO: rendere programmabili alcuni parametri come numero di classi
        modelPath.str(std::string());
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
            CNNFunction::train(epoch, model, device, *train_loader, optimizer, train_dataset_size, parameters);
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
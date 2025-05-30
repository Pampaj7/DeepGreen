#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "CIFAR100.h"


// Where to find the CIFAR10 dataset.
const char* kDataRoot = "/../data/cifar100_png";
const char* kClassesPath = "/../data/cifar100_png/classes.json";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000; //TODO

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


template <typename DataLoader>
void train(
        size_t epoch,
        torch::jit::script::Module& model,
        torch::Device device,
        DataLoader& data_loader,
        torch::optim::Optimizer& optimizer,
        size_t dataset_size,
        std::vector<torch::Tensor> parameters) {

    torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().reduction(torch::kMean));
    model.train();
    size_t batch_idx = 0;
    for (auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        // std::cout << torch::size(data, 0) << ", " << torch::size(data, 1) << ", " << torch::size(data, 2) << ", " << torch::size(data, 3) << std::endl;
        optimizer.zero_grad();
        //assert(!data.isnan().any().item<bool>());
        //assert(!data.isinf().any().item<bool>());
        auto output = model.forward({data}).toTensor();
        /*for (int i = 0; i < targets.size(0); i++)
        {
            std::cout << targets[i].item() << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i < output.size(0); i++)
        {
            for (int j = 0; j < output.size(1); j++)
            {
                std::cout << output[i][j].item() << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Logits max: " << output.max().item<float>() << std::endl;
        std::cout << "Logits min: " << output.min().item<float>() << std::endl;*/

        auto loss = criterion(output, targets);
        // std::cout << "Loss value: " << loss.template item<float>() << std::endl;
        TORCH_INTERNAL_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();

        //auto total_norm = torch::nn::utils::clip_grad_norm_(parameters,50);
        //std::cout << "Gradient norm: " << total_norm << std::endl;
        optimizer.step();

        if (batch_idx++ % kLogInterval == 0) {
            std::printf(
                "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f\n",
                epoch,
                batch_idx * batch.data.size(0),
                dataset_size,
                loss.template item<float>());
        }
    }
}

template <typename DataLoader>
void test(
    torch::jit::script::Module& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
    torch::NoGradGuard no_grad;
    model.eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        auto output = model.forward({data}).toTensor();
        test_loss += torch::nn::functional::cross_entropy(
                         output,
                         targets,
                         torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum))
                         .template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    std::printf(
        "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
        test_loss,
        static_cast<double>(correct) / dataset_size);
}



torch::jit::script::Module load_model(std::string  model_path){
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        throw new std::runtime_error(e.msg());
    }
    std::cout << model_path << " loaded\n";
    return module;
}


int main() {
    try {
        // device (CPU or GPU)
        torch::DeviceType device_type;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Training on GPU." << std::endl;
            device_type = torch::kCUDA;
        } else {
            std::cout << "Training on CPU." << std::endl;
            device_type = torch::kCPU;
        }
        torch::Device device(device_type);
/*
        std::string imageRelativePath = "/../data/test/01014.png";
        std::stringstream imgFullPathStream;
        imgFullPathStream << PROJECT_SOURCE_DIR << imageRelativePath;
        cv::Mat image = cv::imread(imgFullPathStream.str(), cv::IMREAD_COLOR);
        if (image.empty()) {
            CV_Error(-1, "Failed to load image at: " + imgFullPathStream.str());
        }
        imgFullPathStream.str(std::string());
*/




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
        torch::jit::script::Module model = load_model(modelPath.str()); // TODO: rendere programmabili alcuni parametri come numero di classi
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
            train(epoch, model, device, *train_loader, optimizer, train_dataset_size, parameters);
            test(model, device, *test_loader, test_dataset_size);
            //TODO: tracker.stop()
        }

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
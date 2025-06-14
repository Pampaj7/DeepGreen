#ifndef CNNFUNCTION_H
#define CNNFUNCTION_H
#include <torch/torch.h>


// After how many batches to log a new update with the loss value.
constexpr int64_t kLogInterval = 10;

namespace CNNFunction {

    template <typename DataLoader>
    void train(
            const size_t epoch,
            torch::jit::script::Module& model,
            torch::Device device,
            DataLoader& data_loader,
            torch::optim::Optimizer& optimizer,
            const size_t dataset_size,
            torch::nn::CrossEntropyLoss& criterion) {

        criterion->options.reduction(torch::kMean);
        model.train();
        size_t batch_idx = 0;
        for (auto& batch : data_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model.forward({data}).toTensor();
            auto loss = criterion(output, targets);
            TORCH_INTERNAL_ASSERT(!std::isnan(loss.template item<float>()));
            loss.backward();
            optimizer.step();

            if (batch_idx++ % kLogInterval == 0) {
                std::printf(
                    "\rTrain Epoch: %llu [%5zu/%5llu] Loss: %.4f\n",
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
            const size_t dataset_size,
            torch::nn::CrossEntropyLoss& criterion) {
        torch::NoGradGuard no_grad;
        model.eval();
        double test_loss = 0;
        int32_t correct = 0;
        criterion->options.reduction(torch::kSum);

        for (const auto& batch : data_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            auto output = model.forward({data}).toTensor();
            test_loss += criterion(output, targets).template item<float>();
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().template item<int64_t>();
        }

        test_loss /= dataset_size;
        std::printf(
            "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
            test_loss,
            static_cast<double>(correct) / dataset_size);
    }

};



#endif //CNNFUNCTION_H

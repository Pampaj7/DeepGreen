#ifndef CNNFUNCTION_H
#define CNNFUNCTION_H
#include <torch/torch.h>


// After how many batches to log a new update with the loss value.
constexpr int64_t kLogInterval = 10;

namespace CNNFunction {

    /**
     * Training TorchScript model (torch::jit::script::Module)
     */
    template <typename DataLoader>
    float train(
            torch::jit::script::Module& model,
            torch::Device device,
            DataLoader& data_loader,
            torch::optim::Optimizer& optimizer,
            const size_t dataset_size,
            torch::nn::CrossEntropyLoss& criterion) {

        criterion->options.reduction(torch::kMean);
        model.train();

        size_t batch_idx = 0;
        int64_t num_running_corrects = 0;
        int64_t num_samples = 0;
        float running_loss = 0;

        for (auto& batch : data_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model.forward({data}).toTensor();
            auto loss = criterion(output, targets);
            TORCH_INTERNAL_ASSERT(!std::isnan(loss.template item<float>()));

            loss.backward();
            optimizer.step();

            auto pred = output.argmax(1);
            num_running_corrects += pred.eq(targets).sum().template item<int64_t>();
            num_samples += data.size(0);
            running_loss += loss.template item<float>() * data.size(0);

            if (++batch_idx % kLogInterval == 0) {
                std::printf(
                    "\rTraining [%5zu/%5llu] Loss: %.4f | Accuracy: %.3f\n",
                    num_samples,
                    dataset_size,
                    running_loss / num_samples,
                    static_cast<float>(num_running_corrects) / num_samples);
            }
        }

        return running_loss / num_samples;
    }

    /**
     * Inference TorchScript model (torch::jit::script::Module)
     */
    template <typename DataLoader>
    std::array<double, 2> test(
            torch::jit::script::Module& model,
            torch::Device device,
            DataLoader& data_loader,
            const size_t dataset_size,
            torch::nn::CrossEntropyLoss& criterion) {

        torch::NoGradGuard no_grad;
        model.eval();
        criterion->options.reduction(torch::kSum);

        double test_loss = 0;
        int64_t correct = 0;

        for (const auto& batch : data_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            auto output = model.forward({data}).toTensor();

            test_loss += criterion(output, targets).template item<float>();
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().template item<int64_t>();
        }

        return {test_loss /= dataset_size, static_cast<double>(100 * correct) / dataset_size};
    }

    /**
     * Training LibTorch model (torch::nn::ModuleHolder<T>)
     */
    template <typename Model, typename DataLoader>
    float train(
            torch::nn::ModuleHolder<Model>& model,
            torch::Device device,
            DataLoader& data_loader,
            torch::optim::Optimizer& optimizer,
            const size_t dataset_size,
            torch::nn::CrossEntropyLoss& criterion) {

        criterion->options.reduction(torch::kMean);
        model->train();

        // DEBUG ONLY
        // size_t batch_idx = 0;
        int64_t num_running_corrects = 0;
        int64_t num_samples = 0;
        float running_loss = 0;

        for (auto& batch : data_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = criterion(output, targets);
            // DEBUG ONLY
            // TORCH_INTERNAL_ASSERT(!std::isnan(loss.template item<float>()));

            loss.backward();
            optimizer.step();

            auto pred = output.argmax(1);
            num_running_corrects += pred.eq(targets).sum().template item<int64_t>();
            num_samples += data.size(0);
            running_loss += loss.template item<float>() * data.size(0);

            // // DEBUG ONLY
            // if (++batch_idx % kLogInterval == 0) {
            //     std::printf(
            //         "Training [%5llu/%5llu] Loss: %.4f | Accuracy: %.3f\n",
            //         num_samples,
            //         dataset_size,
            //         running_loss / num_samples,
            //         static_cast<float>(num_running_corrects) / num_samples);
            // }
        }

        return running_loss / num_samples;
    }

    /**
     * Inference LibTorch model (torch::nn::ModuleHolder<T>)
     */
    template <typename Model, typename DataLoader>
    std::array<double, 2> test(
            torch::nn::ModuleHolder<Model>& model,
            torch::Device device,
            DataLoader& data_loader,
            const size_t dataset_size,
            torch::nn::CrossEntropyLoss& criterion) {

        torch::NoGradGuard no_grad;
        model->eval();
        criterion->options.reduction(torch::kSum);

        double test_loss = 0;
        int32_t correct = 0;

        for (const auto& batch : data_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            auto output = model->forward(data);

            test_loss += criterion(output, targets).template item<float>();
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().template item<int64_t>();
        }

        return {test_loss /= dataset_size, static_cast<double>(100 * correct) / dataset_size};
    }

};



#endif //CNNFUNCTION_H

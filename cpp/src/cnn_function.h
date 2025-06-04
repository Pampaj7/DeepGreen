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
            const size_t dataset_size) {

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
            optimizer.step();

            if (batch_idx++ % kLogInterval == 0) {
                std::printf(
                    "\rTrain Epoch: %llu [%5ld/%5llu] Loss: %.4f\n",
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
        const size_t dataset_size) {
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

};



#endif //CNNFUNCTION_H

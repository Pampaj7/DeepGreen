#ifndef RESNET18_NATIVE_H
#define RESNET18_NATIVE_H

#include <torch/torch.h>

namespace vision::models
{
    namespace _resnetimpl {
        // 3x3 convolution with padding
        torch::nn::Conv2d conv3x3(int64_t in, int64_t out, int64_t stride = 1, int64_t groups = 1);

        // 1x1 convolution
        torch::nn::Conv2d conv1x1(int64_t in, int64_t out, int64_t stride = 1);

        struct BasicBlock : torch::nn::Module {
            static int expansion;

            int64_t stride;
            torch::nn::Sequential downsample;
            torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

            BasicBlock(
                int64_t inplanes,
                int64_t planes,
                int64_t stride = 1,
                const torch::nn::Sequential& downsample = nullptr,
                int64_t groups = 1,
                int64_t base_width = 64);

            torch::Tensor forward(torch::Tensor x);
        };
    } // namespace _resnetimpl


    struct ResNetImpl : torch::nn::Module {
        int64_t groups, base_width, inplanes;
        torch::nn::Conv2d conv1;
        torch::nn::BatchNorm2d bn1;
        torch::nn::Sequential layer1, layer2, layer3, layer4;
        torch::nn::Linear fc;

        explicit ResNetImpl(
            const std::vector<int64_t>& layers,
            int64_t num_classes = 1000,
            bool zero_init_residual = false,
            int64_t groups = 1,
            int64_t width_per_group = 64);

        torch::nn::Sequential _make_layer(
            int64_t planes,
            int64_t blocks,
            int64_t stride = 1);

        torch::Tensor forward(torch::Tensor x);
    };

    struct ResNet18Impl : ResNetImpl {
        explicit ResNet18Impl(
            int64_t num_classes = 1000,
            bool zero_init_residual = false);
    };

    TORCH_MODULE(ResNet18);

}


#endif //RESNET18_NATIVE_H

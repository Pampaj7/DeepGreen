#ifndef RESNET18_H
#define RESNET18_H

#include <memory>
#include <torch/torch.h>

namespace models
{
    namespace _resnetimpl
    {
        // 3x3 convolution with padding
        torch::nn::Conv2d conv3x3(int64_t in_planes, int64_t out_planes,
                              int64_t stride = 1, int64_t groups = 1, int64_t dilation = 1);

        // 1x1 convolution
        torch::nn::Conv2d conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride = 1);

        struct BasicBlock : torch::nn::Module
        {
            BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
                const torch::nn::Sequential& downsample = nullptr,
                int64_t groups = 1, int64_t base_width = 64, int64_t dilation = 1);

            static constexpr int64_t m_expansion = 1;

            torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr};
            torch::nn::BatchNorm2d m_bn1{nullptr}, m_bn2{nullptr};
            torch::nn::ReLU m_relu{nullptr};
            torch::nn::Sequential m_downsample{nullptr};

            int64_t m_stride;

            torch::Tensor forward(torch::Tensor x);
        };
    } // namespace _resnetimpl


    struct ResNetImpl : torch::nn::Module
    {
        int64_t m_groups, m_base_width, m_inplanes, m_dilation;

        torch::nn::Conv2d m_conv1{nullptr};
        torch::nn::BatchNorm2d m_bn1{nullptr};
        torch::nn::ReLU m_relu{nullptr};
        torch::nn::MaxPool2d m_maxpool{nullptr};
        torch::nn::Sequential m_layer1{nullptr}, m_layer2{nullptr},
            m_layer3{nullptr}, m_layer4{nullptr};
        torch::nn::AdaptiveAvgPool2d m_avgpool{nullptr};
        torch::nn::Linear m_fc{nullptr};

        torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks,
            int64_t stride = 1, bool dilate = false);

        explicit ResNetImpl(const std::vector<int64_t>& layers,
            int64_t num_classes = 1000, bool zero_init_residual = false,
            int64_t groups = 1, int64_t width_per_group = 64,
            std::vector<int64_t> replace_stride_with_dilation = {});

        torch::Tensor forward(torch::Tensor x);
    };

    struct ResNet18Impl final : ResNetImpl
    {
        explicit ResNet18Impl(int64_t num_classes = 1000,
            bool zero_init_residual = false, int64_t groups = 1,
            int64_t width_per_group = 64,
            const std::vector<int64_t>& replace_stride_with_dilation = {});
    };

    TORCH_MODULE(ResNet18);

} // namespace models



#endif //RESNET18_H

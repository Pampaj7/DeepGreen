#ifndef VGG16_H
#define VGG16_H

#include <torch/torch.h>

namespace models
{
    struct VGGImpl : torch::nn::Module
    {
        torch::nn::Sequential m_features{nullptr};
        torch::nn::AdaptiveAvgPool2d m_avgpool{nullptr};
        torch::nn::Sequential m_classifier{nullptr};

        void _initialize_weights();

        explicit VGGImpl(const torch::nn::Sequential& features,
            int64_t num_classes = 1000, bool init_weights = true,
            double dropout = 0.5);

        torch::Tensor forward(torch::Tensor x);
    };

    torch::nn::Sequential makeLayers(const std::vector<int>& cfg);

    struct VGG16Impl final : VGGImpl
    {
        explicit VGG16Impl(int64_t num_classes = 1000, bool init_weights = true);
    };

    TORCH_MODULE(VGG16);

} // namespace models



#endif //VGG16_H

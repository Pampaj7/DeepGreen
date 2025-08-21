#ifndef VGG16_NATIVE_H
#define VGG16_NATIVE_H

#include <torch/torch.h>

namespace vision::models
{
    namespace _vggimpl {
        // TODO here torch::relu_ and torch::adaptive_avg_pool2d wrapped in
        // torch::nn::Fuctional don't work. so keeping these for now

        torch::Tensor& relu_(const torch::Tensor& x);

        torch::Tensor max_pool2d(
            const torch::Tensor& x,
            torch::ExpandingArray<2> kernel_size,
            torch::ExpandingArray<2> stride);

        torch::Tensor adaptive_avg_pool2d(
            const torch::Tensor& x,
            torch::ExpandingArray<2> output_size);

    }// namespace _vggimpl

    struct VGGImpl : torch::nn::Module {
        torch::nn::Sequential features{nullptr}, classifier{nullptr};

        void _initialize_weights();

        explicit VGGImpl(
            const torch::nn::Sequential& features,
            int64_t num_classes = 1000,
            bool initialize_weights = true,
            double dropout = 0.5);

        torch::Tensor forward(torch::Tensor x);
    };

    torch::nn::Sequential makeLayers(const std::vector<int>& cfg);

    // VGG 16-layer model (configuration "D")
    struct VGG16Impl : VGGImpl {
        explicit VGG16Impl(
            int64_t num_classes = 1000,
            bool initialize_weights = true);
    };

    TORCH_MODULE(VGG16);

}


#endif //VGG16_NATIVE_H










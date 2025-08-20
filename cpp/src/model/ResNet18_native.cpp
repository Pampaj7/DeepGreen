#include "ResNet18_native.h"

namespace vision {
    namespace models {
        namespace _resnetimpl {
            torch::nn::Conv2d conv3x3(
                    int64_t in,
                    int64_t out,
                    int64_t stride,
                    int64_t groups) {
                torch::nn::Conv2dOptions O(in, out, 3);
                O.padding(1).stride(stride).groups(groups).bias(false);
                return torch::nn::Conv2d(O);
            }

            torch::nn::Conv2d conv1x1(int64_t in, int64_t out, int64_t stride) {
                torch::nn::Conv2dOptions O(in, out, 1);
                O.stride(stride).bias(false);
                return torch::nn::Conv2d(O);
            }

            int BasicBlock::expansion = 1;

            BasicBlock::BasicBlock(
                int64_t inplanes,
                int64_t planes,
                int64_t stride,
                const torch::nn::Sequential& downsample,
                int64_t groups,
                int64_t base_width)
                : stride(stride), downsample(downsample) {
                TORCH_CHECK(
                    groups == 1 && base_width == 64,
                    "BasicBlock only supports groups=1 and base_width=64");

                // Both conv1 and downsample layers downsample the input when stride != 1
                conv1 = conv3x3(inplanes, planes, stride);
                conv2 = conv3x3(planes, planes);

                bn1 = torch::nn::BatchNorm2d(planes);
                bn2 = torch::nn::BatchNorm2d(planes);

                register_module("conv1", conv1);
                register_module("conv2", conv2);

                register_module("bn1", bn1);
                register_module("bn2", bn2);

                if (!downsample.is_empty())
                    register_module("downsample", this->downsample);
            }

            torch::Tensor BasicBlock::forward(torch::Tensor x) {
                auto identity = x;

                auto out = conv1->forward(x);
                out = bn1->forward(out).relu_();

                out = conv2->forward(out);
                out = bn2->forward(out);

                if (!downsample.is_empty())
                    identity = downsample->forward(x);

                out += identity;
                return out.relu_();
            }
        } // namespace _resnetimpl

        ResNet18Impl::ResNet18Impl(int64_t num_classes, bool zero_init_residual)
            : ResNetImpl({2, 2, 2, 2}, num_classes, zero_init_residual) {}

        std::shared_ptr<ResNet18Impl>
        build_native_resnet18(int64_t num_classes, bool zero_init_residual)
        {
            std::shared_ptr<ResNet18Impl> model = std::make_shared<ResNet18Impl>(num_classes, zero_init_residual);
            return model;
        }

    } // namespace models
} // namespace vision


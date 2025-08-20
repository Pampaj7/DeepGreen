#ifndef RESNET18_NATIVE_H
#define RESNET18_NATIVE_H

#include <torch/torch.h>

namespace vision {
    namespace models {
        struct ResNetImpl;

        namespace _resnetimpl {
            // 3x3 convolution with padding
            torch::nn::Conv2d conv3x3(int64_t in, int64_t out, int64_t stride = 1, int64_t groups = 1);

            // 1x1 convolution
            torch::nn::Conv2d conv1x1(int64_t in, int64_t out, int64_t stride = 1);

            struct BasicBlock : torch::nn::Module {
                friend struct vision::models::ResNetImpl;

                int64_t stride;
                torch::nn::Sequential downsample;

                torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
                torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

                static int expansion;

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

            torch::nn::Sequential _make_layer(
                int64_t planes,
                int64_t blocks,
                int64_t stride = 1);

            explicit ResNetImpl(
                const std::vector<int>& layers,
                int64_t num_classes = 1000,
                bool zero_init_residual = false,
                int64_t groups = 1,
                int64_t width_per_group = 64);

            torch::Tensor forward(torch::Tensor X);
        };

        inline torch::nn::Sequential ResNetImpl::_make_layer(
                int64_t planes,
                int64_t blocks,
                int64_t stride)
        {
            torch::nn::Sequential downsample = nullptr;
            if (stride != 1 || inplanes != planes * _resnetimpl::BasicBlock::expansion) {
            downsample = torch::nn::Sequential(
                _resnetimpl::conv1x1(inplanes, planes * _resnetimpl::BasicBlock::expansion, stride),
                torch::nn::BatchNorm2d(planes * _resnetimpl::BasicBlock::expansion));
            }

            torch::nn::Sequential layers;
            layers->push_back(
                _resnetimpl::BasicBlock(inplanes, planes, stride, downsample, groups, base_width));

            inplanes = planes * _resnetimpl::BasicBlock::expansion;

            for (int i = 1; i < blocks; ++i)
                layers->push_back(_resnetimpl::BasicBlock(inplanes, planes, 1, nullptr, groups, base_width));

            return layers;
        }

        inline ResNetImpl::ResNetImpl(
            const std::vector<int>& layers,
            int64_t num_classes,
            bool zero_init_residual,
            int64_t groups,
            int64_t width_per_group)
            :   groups(groups),
                base_width(width_per_group),
                inplanes(64),
                conv1(
                  torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)),
                bn1(64),
                layer1(_make_layer(64, layers[0])),
                layer2(_make_layer(128, layers[1], 2)),
                layer3(_make_layer(256, layers[2], 2)),
                layer4(_make_layer(512, layers[3], 2)),
                fc(512 * _resnetimpl::BasicBlock::expansion, num_classes) {
            register_module("conv1", conv1);
            register_module("bn1", bn1);
            register_module("fc", fc);

            register_module("layer1", layer1);
            register_module("layer2", layer2);
            register_module("layer3", layer3);
            register_module("layer4", layer4);

            for (auto& module : modules(/*include_self=*/false)) {
                if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
                    torch::nn::init::kaiming_normal_(
                        M->weight,
                        /*a=*/0,
                        torch::kFanOut,
                        torch::kReLU);
                else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
                    torch::nn::init::constant_(M->weight, 1);
                    torch::nn::init::constant_(M->bias, 0);
                }
            }

            // Zero-initialize the last BN in each residual branch, so that the residual
            // branch starts with zeros, and each residual block behaves like an
            // identity. This improves the model by 0.2~0.3% according to
            // https://arxiv.org/abs/1706.02677
            if (zero_init_residual)
                for (auto& module : modules(/*include_self=*/false)) {
                    if (auto* M = dynamic_cast<_resnetimpl::BasicBlock*>(module.get()))
                        torch::nn::init::constant_(M->bn2->weight, 0);
                }
        }

        inline torch::Tensor ResNetImpl::forward(torch::Tensor x) {
            x = conv1->forward(x);
            x = bn1->forward(x).relu_();
            x = torch::max_pool2d(x, 3, 2, 1);

            x = layer1->forward(x);
            x = layer2->forward(x);
            x = layer3->forward(x);
            x = layer4->forward(x);

            x = torch::adaptive_avg_pool2d(x, {1, 1});
            x = x.reshape({x.size(0), -1});
            x = fc->forward(x);

            return x;
        }

        struct ResNet18Impl : ResNetImpl {
            explicit ResNet18Impl(
                int64_t num_classes = 1000,
                bool zero_init_residual = false);
        };

        std::shared_ptr<ResNet18Impl>
        build_native_resnet18(int64_t num_classes = 1000, bool zero_init_residual = false);

        TORCH_MODULE(ResNet18);

    } // namespace models
} // namespace vision




#endif //RESNET18_NATIVE_H

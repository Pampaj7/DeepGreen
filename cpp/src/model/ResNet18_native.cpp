#include "ResNet18_native.h"

torch::nn::Conv2d vision::models::_resnetimpl::conv3x3(
        int64_t in,
        int64_t out,
        int64_t stride,
        int64_t groups) {
    torch::nn::Conv2dOptions O(in, out, 3);
    O.padding(1).stride(stride).groups(groups).bias(false);
    return torch::nn::Conv2d(O);
}

torch::nn::Conv2d vision::models::_resnetimpl::conv1x1(int64_t in, int64_t out, int64_t stride) {
    torch::nn::Conv2dOptions O(in, out, 1);
    O.stride(stride).bias(false);
    return torch::nn::Conv2d(O);
}

int vision::models::_resnetimpl::BasicBlock::expansion = 1;

vision::models::_resnetimpl::BasicBlock::BasicBlock(
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

torch::Tensor vision::models::_resnetimpl::BasicBlock::forward(torch::Tensor x) {
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



torch::nn::Sequential vision::models::ResNetImpl::_make_layer(
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

vision::models::ResNetImpl::ResNetImpl(
        const std::vector<int64_t>& layers,
        int64_t num_classes,
        bool zero_init_residual,
        int64_t groups,
        int64_t width_per_group)
: groups(groups), base_width(width_per_group), inplanes(64),
conv1(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)),
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

torch::Tensor vision::models::ResNetImpl::forward(torch::Tensor x) {
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

vision::models::ResNet18Impl::ResNet18Impl(int64_t num_classes, bool zero_init_residual)
: ResNetImpl({2, 2, 2, 2}, num_classes, zero_init_residual) {}

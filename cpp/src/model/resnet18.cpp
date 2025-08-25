#include "resnet18.h"

#include <memory>
#include <vector>
#include <torch/torch.h>

torch::nn::Conv2dOptions
create_conv2d_options(const int64_t in_planes, const int64_t out_planes,
        const int64_t kerner_size, const int64_t stride = 1,
        const int64_t padding = 0, const int64_t groups = 1,
        const int64_t dilation = 1, const bool bias = false)
{
    const torch::nn::Conv2dOptions conv_options =
        torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size)
            .stride(stride)
            .padding(padding)
            .bias(bias)
            .groups(groups)
            .dilation(dilation);

    return conv_options;
}

torch::nn::Conv2d models::_resnetimpl::conv3x3(
        const int64_t in_planes,
        const int64_t out_planes,
        const int64_t stride,
        const int64_t groups,
        const int64_t dilation)
{
    torch::nn::Conv2dOptions conv_options = create_conv2d_options(
        in_planes, out_planes, /*kerner_size = */ 3, stride,
        /*padding = */ dilation, groups, /*dilation = */ dilation, false);
    return torch::nn::Conv2d(conv_options);
}

torch::nn::Conv2d models::_resnetimpl::conv1x1(
        const int64_t in_planes,
        const int64_t out_planes,
        const int64_t stride)
{
    torch::nn::Conv2dOptions conv_options = create_conv2d_options(
        in_planes, out_planes, /*kerner_size = */ 1, stride,
        /*padding = */ 0, /*groups = */ 1, /*dilation = */ 1, false);
    return torch::nn::Conv2d(conv_options);
}



models::_resnetimpl::BasicBlock::BasicBlock(
        const int64_t inplanes,
        const int64_t planes,
        const int64_t stride,
        const torch::nn::Sequential& downsample,
        const int64_t groups,
        const int64_t base_width,
        const int64_t dilation)
{
    TORCH_CHECK(
        groups == 1 && base_width == 64,
        "BasicBlock only supports groups=1 and base_width=64");
    TORCH_CHECK(
            dilation <= 1,
            "Dilation > 1 not supported in BasicBlock");

    // Both conv1 and downsample layers downsample the input when stride != 1
    m_conv1 = register_module("conv1", conv3x3(inplanes, planes, stride));
    m_bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
    m_relu = register_module("relu", torch::nn::ReLU(true));
    m_conv2 = register_module("conv2", conv3x3(planes, planes));
    m_bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));

    if (!downsample.is_empty())
        m_downsample = register_module("downsample", downsample);
    m_stride = stride;
}

torch::Tensor models::_resnetimpl::BasicBlock::forward(torch::Tensor x)
{
    auto identity = x;

    auto out = m_conv1->forward(x);
    out = m_bn1->forward(out);
    out = m_relu->forward(out);

    out = m_conv2->forward(out);
    out = m_bn2->forward(out);

    if (!m_downsample.is_empty())
        identity = m_downsample->forward(x);

    out += identity;
    out = m_relu->forward(out);

    return out;
}



torch::nn::Sequential models::ResNetImpl::_make_layer(
        const int64_t planes,
        const int64_t blocks,
        int64_t stride,
        const bool dilate)
{
    torch::nn::Sequential downsample = nullptr;
    const int64_t previous_dilation = m_dilation;
    if (dilate)
    {
        m_dilation *= stride;
        stride = 1;
    }
    if (stride != 1 || m_inplanes != planes * _resnetimpl::BasicBlock::m_expansion) {
        downsample = torch::nn::Sequential(_resnetimpl::conv1x1(
            m_inplanes, planes * _resnetimpl::BasicBlock::m_expansion, stride),
            torch::nn::BatchNorm2d(planes * _resnetimpl::BasicBlock::m_expansion));
    }

    torch::nn::Sequential layers;
    layers->push_back(
        _resnetimpl::BasicBlock(m_inplanes, planes, stride, downsample,
                                m_groups, m_base_width, previous_dilation));

    m_inplanes = planes * _resnetimpl::BasicBlock::m_expansion;

    for (int64_t i = 1; i < blocks; ++i)
        layers->push_back(
            _resnetimpl::BasicBlock(m_inplanes, planes, 1,
                nullptr, m_groups, m_base_width, m_dilation));

    return layers;
}

models::ResNetImpl::ResNetImpl(
        const std::vector<int64_t>& layers,
        const int64_t num_classes,
        const bool zero_init_residual,
        const int64_t groups,
        const int64_t width_per_group,
        std::vector<int64_t> replace_stride_with_dilation)
    : m_groups(groups),
      m_base_width(width_per_group),
      m_inplanes(64),
      m_dilation(1)
{
    if (replace_stride_with_dilation.size() == 0)
        // Each element in the tuple indicates if we should replace
        // the 2x2 stride with a dilated convolution instead.
        replace_stride_with_dilation = {false, false, false};
    TORCH_CHECK(
        replace_stride_with_dilation.size() == 3,
        "replace_stride_with_dilation should be empty or have exactly three elements.");

    m_conv1 = register_module("conv1",
        torch::nn::Conv2d(create_conv2d_options(/*in_planes = */ 3, /*out_planes = */ m_inplanes,
                                            /*kerner_size = */ 7, /*stride = */ 2, /*padding = */ 3,
                                            /*groups = */ 1, /*dilation = */ 1, /*bias = */ false)));
    m_bn1 = register_module("bn1", torch::nn::BatchNorm2d(m_inplanes));
    m_relu = register_module("relu", torch::nn::ReLU(true));
    m_maxpool = register_module("maxpool", torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1})));

    m_layer1 = register_module("layer1", _make_layer(64, layers.at(0)));
    m_layer2 = register_module("layer2", _make_layer(128, layers.at(1), 2,
                              replace_stride_with_dilation.at(0)));
    m_layer3 = register_module("layer3", _make_layer(256, layers.at(2), 2,
                              replace_stride_with_dilation.at(1)));
    m_layer4 = register_module("layer4", _make_layer(512, layers.at(3), 2,
                              replace_stride_with_dilation.at(2)));

    m_avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(
                       torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    m_fc = register_module("fc", torch::nn::Linear(512 * _resnetimpl::BasicBlock::m_expansion, num_classes));

    for (auto& module : modules(/*include_self=*/false)) {
        if (const auto conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
            torch::nn::init::kaiming_normal_(
                conv->weight,
                /*a=*/0,
                torch::kFanOut,
                torch::kReLU);
        else if (const auto bn = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
            torch::nn::init::constant_(bn->weight, 1);
            torch::nn::init::constant_(bn->bias, 0);
        }
    }

    // Zero-initialize the last BN in each residual branch, so that the residual
    // branch starts with zeros, and each residual block behaves like an
    // identity. This improves the model by 0.2~0.3% according to
    // https://arxiv.org/abs/1706.02677
    if (zero_init_residual)
        for (auto& module : modules(/*include_self=*/false)) {
            if (auto* M = dynamic_cast<_resnetimpl::BasicBlock*>(module.get()))
                torch::nn::init::constant_(M->m_bn2->weight, 0);
        }
}

torch::Tensor models::ResNetImpl::forward(torch::Tensor x)
{
    x = m_conv1->forward(x);
    x = m_bn1->forward(x);
    x = m_relu->forward(x);
    x = m_maxpool->forward(x);

    x = m_layer1->forward(x);
    x = m_layer2->forward(x);
    x = m_layer3->forward(x);
    x = m_layer4->forward(x);

    x = m_avgpool->forward(x);
    x = torch::flatten(x, 1);
    x = m_fc->forward(x);

    return x;
}

models::ResNet18Impl::ResNet18Impl(
        const int64_t num_classes,
        const bool zero_init_residual,
        const int64_t groups,
        const int64_t width_per_group,
        const std::vector<int64_t>& replace_stride_with_dilation)
    : ResNetImpl({2, 2, 2, 2}, num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation) {}

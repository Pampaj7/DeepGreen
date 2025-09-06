#include "vgg16.h"

void models::VGGImpl::_initialize_weights()
{
    for (auto& module : modules(/*include_self=*/false)) {
        if (const auto conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
            torch::nn::init::kaiming_normal_(
                conv->weight,
                /*a=*/0,
                torch::kFanOut,
                torch::kReLU);
            // Here there is no if statement (wrt PyTorch) because torch::nn::Conv2dImpl always has the bias field
            // If it is defined with bias(false), then init::constant_ does nothing.
            torch::nn::init::constant_(conv->bias, 0);
        } else if (const auto linear = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
            torch::nn::init::normal_(linear->weight, 0, 0.01);
            torch::nn::init::constant_(linear->bias, 0);
        }
    }
}

models::VGGImpl::VGGImpl(
    torch::nn::Sequential features,
    const int64_t num_classes,
    const bool init_weights,
    const double dropout)
{
    m_features = register_module("features", std::move(features));
    m_avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(
                       torch::nn::AdaptiveAvgPool2dOptions({7, 7})));
    m_classifier = register_module("classifier", torch::nn::Sequential(
        torch::nn::Linear(512 * 7 * 7, 4096),
            torch::nn::ReLU(true),
            torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout)),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU(true),
            torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout)),
            torch::nn::Linear(4096, num_classes)
        )
    );

    // basic weights initialization
    if (init_weights)
        _initialize_weights();
}

torch::Tensor models::VGGImpl::forward(torch::Tensor x) {
    x = m_features->forward(x);
    x = m_avgpool->forward(x);
    x = torch::flatten(x, 1);
    x = m_classifier->forward(x);
    return x;
}

torch::nn::Sequential models::makeLayers(const std::vector<int>& cfg)
{
    torch::nn::Sequential seq;
    auto in_channels = 3;

    for (const auto& v : cfg) {
        if (v == -1)
            seq->push_back(torch::nn::MaxPool2d(
                torch::nn::MaxPool2dOptions({2, 2}).stride({2, 2})));
        else {
            seq->push_back(torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, v, /* kernel_size = */3).padding(1)));
            seq->push_back(torch::nn::ReLU(true));

            in_channels = v;
        }
    }
    return seq;
}

// VGG 16-layer model (configuration "D")
static std::vector<int> cfg = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};

models::VGG16Impl::VGG16Impl(
        const int64_t num_classes)
    : VGGImpl(makeLayers(cfg), num_classes) {}

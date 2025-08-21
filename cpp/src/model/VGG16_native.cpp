#include "VGG16_native.h"

#include <unordered_map>


torch::Tensor& vision::models::_vggimpl::relu_(const torch::Tensor& x) {
    return x.relu_();
}

torch::Tensor vision::models::_vggimpl::max_pool2d(
            const torch::Tensor& x,
            torch::ExpandingArray<2> kernel_size,
            torch::ExpandingArray<2> stride) {
    return torch::max_pool2d(x, kernel_size, stride);
}

torch::Tensor vision::models::_vggimpl::adaptive_avg_pool2d(
            const torch::Tensor& x,
            torch::ExpandingArray<2> output_size) {
    return torch::adaptive_avg_pool2d(x, output_size);
}

void vision::models::VGGImpl::_initialize_weights() {
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
            torch::nn::init::kaiming_normal_(
                M->weight,
                /*a=*/0,
                torch::kFanOut,
                torch::kReLU);
            // Here there is no if statement (wrt PyTorch) because torch::nn::Conv2dImpl always has the bias field
            // If it is defined with bias(false), then init::constant_ does nothing.
            torch::nn::init::constant_(M->bias, 0);
        } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
            torch::nn::init::normal_(M->weight, 0, 0.01);
            torch::nn::init::constant_(M->bias, 0);
        }
    }
}

vision::models::VGGImpl::VGGImpl(
    const torch::nn::Sequential& features,
    int64_t num_classes,
    bool initialize_weights,
    double dropout) //TODO: usato in PyTorch ma non LibTorch
{
    this->features = features;
    //TODO: avgpool usato inline nel forward
    this->classifier = torch::nn::Sequential(
        torch::nn::Linear(512 * 7 * 7, 4096),
        torch::nn::Functional(_vggimpl::relu_), //TODO: possibile usare solo relu_?
        torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout)),
        torch::nn::Linear(4096, 4096),
        torch::nn::Functional(_vggimpl::relu_), //TODO: possibile usare solo relu_?
        torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout)),
        torch::nn::Linear(4096, num_classes));

    register_module("features", this->features);
    register_module("classifier", classifier);

    if (initialize_weights)
        _initialize_weights();
}

torch::Tensor vision::models::VGGImpl::forward(torch::Tensor x) {
    x = features->forward(x);
    x = _vggimpl::adaptive_avg_pool2d(x, {7, 7});
    x = x.view({x.size(0), -1});
    x = classifier->forward(x);
    return x;
}

torch::nn::Sequential vision::models::makeLayers(const std::vector<int>& cfg)
{
    torch::nn::Sequential seq;
    auto channels = 3; // in_channels

    for (const auto& V : cfg) {
        if (V <= -1)
            seq->push_back(torch::nn::Functional(_vggimpl::max_pool2d, 2, 2)); //TODO: in seq non posso aggiungere direttamente maxpool?
        else {
            seq->push_back(torch::nn::Conv2d(
                torch::nn::Conv2dOptions(channels, V, 3).padding(1)));

            seq->push_back(torch::nn::Functional(_vggimpl::relu_)); //TODO: in seq non posso aggiungere direttamente relu?

            channels = V;
        }
    }

    return seq;
}

// clang-format off
static std::unordered_map<char, std::vector<int>> cfgs = {
    {'D', {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1}}};
// clang-format on

vision::models::VGG16Impl::VGG16Impl(int64_t num_classes, bool initialize_weights)
: VGGImpl(makeLayers(cfgs['D']), num_classes, initialize_weights) {}

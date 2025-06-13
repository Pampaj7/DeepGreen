#include "dataset_transforms.h"

DatasetTransforms::Compose::Compose(std::vector<TorchTrasformPtr> transforms)
        : transforms_(std::move(transforms)) {}

torch::Tensor DatasetTransforms::Compose::operator()(torch::Tensor input)
{
        for (const auto& t : transforms_)
                input = (*t)(input);
        return input;
}


torch::Tensor DatasetTransforms::ReplicateChannels::operator()(const torch::Tensor input)
{
        return input.repeat({3, 1, 1});
}


DatasetTransforms::ResizeTo::ResizeTo(const int64_t target_height, const int64_t target_width)
        : height(target_height), width(target_width) {}

torch::Tensor DatasetTransforms::ResizeTo::operator()(torch::Tensor input)
{
        // Assert that input is as [C, H, W]
        input = input.unsqueeze(0); // -> [1, C, H, W]

        input = torch::nn::functional::interpolate(
            input,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({height, width}))
                .mode(torch::kBilinear)
                .align_corners(false)
        );

        return input.squeeze(0); // -> [C, H, W]
}

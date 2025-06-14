#ifndef DATASET_TRANSFORMS_H
#define DATASET_TRANSFORMS_H
#include <torch/torch.h>

using TorchTransform = torch::data::transforms::TensorTransform<torch::Tensor>;
using TorchTrasformPtr = std::shared_ptr<TorchTransform>;


/**
 * Define custom dataset transformations usable by Torch.
 */
namespace DatasetTransforms {

    /**
     * Collect multiple Torch transformations.
     */
    struct Compose final : public TorchTransform {
        explicit Compose(std::vector<TorchTrasformPtr> transforms);

        torch::Tensor operator()(torch::Tensor input) override;

    private:
        std::vector<TorchTrasformPtr> transforms_;
    };


    /**
     * Replicate images channel from 1 (grayscale) to 3 (RGB).
     */
    struct ReplicateChannels final : public torch::data::transforms::TensorTransform<torch::Tensor> {
        torch::Tensor operator()(torch::Tensor input) override;
    };


    /**
     * Resize images size.
     */
    struct ResizeTo final : public torch::data::transforms::TensorTransform<torch::Tensor> {
        ResizeTo(int64_t target_height, int64_t target_width);

        torch::Tensor operator()(torch::Tensor input) override;

    private:
        int64_t height;
        int64_t width;
    };

};



#endif //DATASET_TRANSFORMS_H

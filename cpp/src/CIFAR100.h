#ifndef CIFAR100_H
#define CIFAR100_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include "cnn_function.h"

using json = nlohmann::json;

class CIFAR100 final : public torch::data::datasets::Dataset<CIFAR100> {
public:
    explicit CIFAR100(const std::string& dataset_path, const std::string& classes_json_path, bool train = false);

    torch::data::Example<> get(size_t index) override;
    [[nodiscard]] torch::optional<size_t> size() const override;

    [[nodiscard]] bool is_train() const noexcept { return train_; }
    // Returns all images stacked into a single tensor.
    [[nodiscard]] const torch::Tensor& images() const { return images_; }
    [[nodiscard]] const torch::Tensor& targets() const { return targets_; }

    static const std::map<std::string, int>& loadClassesToIndexMap(const std::string& path);
    static c10::ArrayRef<double> getMean() { return mean; }
    static c10::ArrayRef<double> getStd() { return std; }

private:
    bool train_;
    torch::Tensor images_, targets_;

    static constexpr uint32_t num_train_samples{50000};
    static constexpr uint32_t num_test_samples{10000};
    static constexpr uint32_t image_height{32};
    static constexpr uint32_t image_width{32};
    static constexpr uint32_t image_channels{3};
    static constexpr std::array<double, 3> mean{0.4914, 0.4822, 0.4465};
    static constexpr std::array<double, 3> std{0.2470, 0.2434, 0.2616};

};



#endif //CIFAR100_H

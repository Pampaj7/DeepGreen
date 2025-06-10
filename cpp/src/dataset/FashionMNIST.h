#ifndef FASHIONMNIST_H
#define FASHIONMNIST_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


class FashionMNIST final : public torch::data::datasets::Dataset<FashionMNIST> {
public:
    explicit FashionMNIST(const std::string& dataset_path, const std::string& classes_json_path, bool train = false);

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

    static constexpr uint32_t num_train_samples{60000};
    static constexpr uint32_t num_test_samples{10000};
    static constexpr uint32_t image_height{28};
    static constexpr uint32_t image_width{28};
    static constexpr uint32_t image_channels{1};
    static constexpr std::array<double, image_channels> mean{0.1307}; //TODO: used MNIST values
    static constexpr std::array<double, image_channels> std{0.3081}; //TODO: used MNIST values

};



#endif //FASHIONMNIST_H

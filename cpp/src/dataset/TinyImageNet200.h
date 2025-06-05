#ifndef TINYIMAGENET200_H
#define TINYIMAGENET200_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


class TinyImageNet200 final : public torch::data::datasets::Dataset<TinyImageNet200> {
public:
    explicit TinyImageNet200(const std::string& dataset_path, const std::string& classes_json_path, bool train = false);

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

    static constexpr uint32_t num_train_samples{100000};
    static constexpr uint32_t num_val_samples{10000}; //TODO
    static constexpr uint32_t num_test_samples{10000}; //TODO
    static constexpr uint32_t image_height{64};
    static constexpr uint32_t image_width{64};
    static constexpr uint32_t image_channels{3};
    static constexpr std::array<double, 3> mean{0.4914, 0.4822, 0.4465}; //TODO
    static constexpr std::array<double, 3> std{0.2470, 0.2434, 0.2616}; //TODO
};



#endif //TINYIMAGENET200_H

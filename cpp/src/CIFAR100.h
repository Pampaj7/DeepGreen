#ifndef CIFAR100_H
#define CIFAR100_H
#include <torch/torch.h>
#include "../tools/json_reader//json.hpp"

using json = nlohmann::json;

class CIFAR100 final : public torch::data::datasets::Dataset<CIFAR100> {
public:
    explicit CIFAR100(const std::string& root, const std::string& classes_json_path, bool train = false);

    torch::data::Example<> get(size_t index) override;

    [[nodiscard]] torch::optional<size_t> size() const override;

    [[nodiscard]] bool is_train() const noexcept;

    // Returns all images stacked into a single tensor.
    [[nodiscard]] const torch::Tensor& images() const;
    [[nodiscard]] const torch::Tensor& targets() const;

    [[nodiscard]] c10::ArrayRef<double> getMean() const;
    [[nodiscard]] c10::ArrayRef<double> getStd() const;

    static const std::map<std::string, int>& loadClassesToIndexMap(const std::string& path);

private:
    bool train_;
    std::map<std::string, int> class_to_index_;
    torch::Tensor images_, targets_;
};



#endif //CIFAR100_H

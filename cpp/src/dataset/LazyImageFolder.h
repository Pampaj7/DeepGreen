#ifndef LAZYIMAGEFOLDER_H
#define LAZYIMAGEFOLDER_H
#include <torch/torch.h>

#include <cassert>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "utils.h"


template <typename Dataset>
class LazyImageFolder final : public torch::data::datasets::Dataset<LazyImageFolder<Dataset>> {
public:
    explicit LazyImageFolder(const std::string& dataset_path, const std::string& classes_json_path, bool train = false);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override {
        return samples_.size();
    }
    bool is_train() const noexcept { return train_; }

private:
    bool train_;
    std::vector<std::pair<std::string, int64_t>> samples_; // {path image, label}
};

template <typename Dataset>
LazyImageFolder<Dataset>::LazyImageFolder(const std::string& dataset_path, const std::string& classes_json_path, bool train)
    : train_(train)
{
    auto class_to_index = Dataset::loadClassesToIndexMap(
        Utils::makeWindowsLongPathIfNeeded(classes_json_path));

    std::string dataset_file_name;
    uint32_t num_samples_per_file;
    if (train_) {
        dataset_file_name = Dataset::getTrainFolder();
        num_samples_per_file = Dataset::getNumTrainSamples();
    } else {
        dataset_file_name = Dataset::getTestFolder();
        num_samples_per_file = Dataset::getNumTestSamples();
    }
    const std::string data_set_file_path = Utils::join_paths(dataset_path, dataset_file_name);

    for (const auto& [class_name, label] : class_to_index) {
        std::string class_path = Utils::join_paths(data_set_file_path, class_name);

        for (auto& img_path : std::filesystem::directory_iterator(class_path)) {
            samples_.emplace_back(
                Utils::makeWindowsLongPathIfNeeded(img_path.path().string()),
                label
            );
        }
    }

    assert(samples_.size() == num_samples_per_file &&
        "Insufficient number of image/label pairs. Data files might have been corrupted.");
}

template <typename Dataset>
torch::data::Example<> LazyImageFolder<Dataset>::get(const size_t index) {
    const auto& [img_str_path, label] = samples_.at(index);

    cv::Mat img = cv::imread(img_str_path,
        Dataset::isGrayscale() ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error("Failed to load image: " + img_str_path);

    if (!Dataset::isGrayscale())
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Normalize to [0,1]
    img.convertTo(img, CV_32F, 1.0f / 255.0f);

    // Convert from HWC to CHW and then to torch::Tensor
    auto img_tensor = torch::from_blob(img.data,
                                   {img.rows, img.cols, img.channels()},
                                   torch::kFloat32)
                    .permute({2, 0, 1})
                    .clone();

    return {img_tensor, torch::tensor(label, torch::kInt64)};
}



#endif //LAZYIMAGEFOLDER_H

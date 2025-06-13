#ifndef IMAGEFOLDER_H
#define IMAGEFOLDER_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <cassert>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "utils.h"


template <typename Dataset>
class ImageFolder final : public torch::data::datasets::Dataset<ImageFolder<Dataset>> {
public:
    explicit ImageFolder(const std::string& dataset_path, const std::string& classes_json_path, bool train = false);

    torch::data::Example<> get(size_t index) override
    {
        return {images_[index], targets_[index]};
    }
    [[nodiscard]] torch::optional<size_t> size() const override
    {
        return images_.size(0);
    }

    [[nodiscard]] bool is_train() const noexcept { return train_; }
    // Returns all images stacked into a single tensor.
    [[nodiscard]] const torch::Tensor& images() const { return images_; }
    [[nodiscard]] const torch::Tensor& targets() const { return targets_; }

private:
    bool train_;
    torch::Tensor images_, targets_;
};


template <typename Dataset>
ImageFolder<Dataset>::ImageFolder(const std::string& dataset_path, const std::string& classes_json_path, bool train)
: train_(train)
{
    auto class_to_index  = Dataset::loadClassesToIndexMap(
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

    std::vector<torch::Tensor> images;
    images.reserve(num_samples_per_file);

    std::vector<int64_t> labels;
    labels.reserve(num_samples_per_file);


    for (const auto& [class_name, label] : class_to_index) {
        std::string class_path = Utils::join_paths(data_set_file_path, class_name);

        for (const auto& img_path : std::filesystem::directory_iterator(class_path)) {
            std::string img_str_path = Utils::makeWindowsLongPathIfNeeded(img_path.path().string());

            cv::Mat img = cv::imread(img_str_path,
                Dataset::isGrayscale() ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR); //TODO: opencv 4.12.0 usa cv::IMREAD_COLOR_BGR
            if (img.empty()) {
                throw std::runtime_error("Failed to load image: " + img_path.path().string());
            }
            if (!Dataset::isGrayscale())
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

            //cv::resize(img, img, cv::Size(32, 32));  // Ensure correct size
            img.convertTo(img, CV_32F, 1.0f / 255.0f); // Normalize to [0,1]

            // Convert from HWC to CHW and then to torch::Tensor
            auto img_tensor = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, torch::kFloat32); //, torch::kUInt8);//
            img_tensor = img_tensor.permute({2, 0, 1}).clone(); // Make it contiguous

            images.push_back(img_tensor);
            labels.push_back(label);
        }
    }

    assert(images.size() == num_samples_per_file &&
        "Insufficient number of images. Data files might have been corrupted.");
    images_ = torch::stack(images); //.to(torch::kFloat32).div_(255)

    assert(labels.size() == num_samples_per_file &&
        "Insufficient number of labels. Data files might have been corrupted.");
    targets_ = torch::tensor(labels, torch::kInt64);
}


#endif //IMAGEFOLDER_H

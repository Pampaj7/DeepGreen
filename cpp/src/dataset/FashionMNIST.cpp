#include "FashionMNIST.h"

#include <cassert>
#include <fstream>
#include <string>
#include <torch/torch.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "utils.h"


FashionMNIST::FashionMNIST(const std::string& dataset_path, const std::string& classes_json_path, const bool train) : train_(train)
{
    auto class_to_index  = loadClassesToIndexMap(classes_json_path);

    std::string dataset_file_name;
    uint32_t num_samples_per_file;
    if (train_) {
        dataset_file_name = "train";
        num_samples_per_file = num_train_samples;
    } else {
        dataset_file_name = "test";
        num_samples_per_file = num_test_samples;
    }
    const std::string data_set_file_path = Utils::join_paths(dataset_path, dataset_file_name);


    std::vector<torch::Tensor> images;
    images.reserve(num_samples_per_file);

    std::vector<int64_t> labels;
    labels.reserve(num_samples_per_file);

    for (const auto& [class_name, label] : class_to_index) {
        std::string class_path = Utils::join_paths(data_set_file_path, class_name);

        for (const auto& img_path : std::filesystem::directory_iterator(class_path)) {
            cv::Mat img = cv::imread(img_path.path().string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                throw std::runtime_error("Failed to load image: " + img_path.path().string());
            }
            //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

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


torch::data::Example<> FashionMNIST::get(const size_t index)
{
    return {images_[index], targets_[index]};
}


torch::optional<size_t> FashionMNIST::size() const
{
    return images_.size(0); //TODO: cos'è size? # di immagini del train/test set o altro?
}


const std::map<std::string, int>& FashionMNIST::loadClassesToIndexMap(const std::string& path)
{
    static std::map<std::string, int> class_to_index;
    static std::once_flag load_flag;

    std::call_once(load_flag, [&]() {
        std::ifstream json_file(path);
        if (!json_file.is_open()) {
            throw std::runtime_error("Unable to open the JSON file at: " + path);
        }

        json class_json;
        json_file >> class_json;

        for (auto& [key, value] : class_json.items())
            class_to_index[value] = std::stoi(key);
        assert(class_to_index.size() == std::stoi(FASHION_MNIST_NUM_CLASSES)); // TODO: in realtà il check dev'essere sull'indice più grande inserito che non deve superare il numero di classi
    });

    return class_to_index;
}
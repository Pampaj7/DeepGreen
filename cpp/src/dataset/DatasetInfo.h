#ifndef DATASETINFO_H
#define DATASETINFO_H
#include <torch/torch.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


template <typename Derived>
class DatasetInfo {
public:
    static uint32_t getNumTrainSamples() { return Derived::num_train_samples; }
    static uint32_t getNumTestSamples() { return Derived::num_test_samples; }
    static uint32_t getImageHeight() { return Derived::image_height; }
    static uint32_t getImageWidth() { return Derived::image_width; }
    static uint32_t getImageChannels() { return Derived::image_channels; }
    static c10::ArrayRef<double> getMean() { return Derived::mean; }
    static c10::ArrayRef<double> getStd() { return Derived::std; }

    static std::string getTrainFolder() { return Derived::train_folder; }
    static std::string getTestFolder() { return Derived::test_folder; }

    static const std::map<std::string, int>& loadClassesToIndexMap(const std::string& path)
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

            int idx_label = 0;
            for (auto& [key, value] : class_json.items())
                class_to_index[value] = idx_label++;
            assert(idx_label == std::stoi(Derived::num_classes)); // required: 0 <= label < num_classes
    });

    return class_to_index;
  }
};


#endif //DATASETINFO_H

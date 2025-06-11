#ifndef TINYIMAGENET200INFO_H
#define TINYIMAGENET200INFO_H
#include <torch/torch.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


class TinyImageNet200Info {
public:
    static uint32_t getNumTrainSamples() { return num_train_samples; }
    static uint32_t getNumTestSamples() { return num_test_samples; }
    static uint32_t getImageHeight() { return image_height; }
    static uint32_t getImageWidth() { return image_width; }
    static uint32_t getImageChannels() { return image_channels; }
    static c10::ArrayRef<double> getMean() { return mean; }
    static c10::ArrayRef<double> getStd() { return std; }

    static std::string getTrainFolder() { return "train"; }
    static std::string getTestFolder() { return "val"; }

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
            assert(class_to_index.size() == std::stoi(TINY_IMAGENET200_NUM_CLASSES));
        });

        return class_to_index;
    }

private:
    static constexpr uint32_t num_train_samples{100000};
    static constexpr uint32_t num_val_samples{10000}; //TODO
    static constexpr uint32_t num_test_samples{10000}; //TODO
    static constexpr uint32_t image_height{64};
    static constexpr uint32_t image_width{64};
    static constexpr uint32_t image_channels{3};
    static constexpr std::array<double, image_channels> mean{0.4914, 0.4822, 0.4465}; //TODO
    static constexpr std::array<double, image_channels> std{0.2470, 0.2434, 0.2616}; //TODO
};



#endif //TINYIMAGENET200INFO_H

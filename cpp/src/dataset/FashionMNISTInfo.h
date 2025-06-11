#ifndef FASHIONMNISTINFO_H
#define FASHIONMNISTINFO_H
#include <torch/torch.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;



class FashionMNISTInfo {
public:
    static uint32_t getNumTrainSamples() { return num_train_samples; }
    static uint32_t getNumTestSamples() { return num_test_samples; }
    static uint32_t getImageHeight() { return image_height; }
    static uint32_t getImageWidth() { return image_width; }
    static uint32_t getImageChannels() { return image_channels; }
    static c10::ArrayRef<double> getMean() { return mean; }
    static c10::ArrayRef<double> getStd() { return std; }

    static std::string getTrainFolder() { return "train"; }
    static std::string getTestFolder() { return "test"; }

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
            assert(class_to_index.size() == std::stoi(FASHION_MNIST_NUM_CLASSES)); // TODO: in realtà il check dev'essere sull'indice più grande inserito che non deve superare il numero di classi
        });

        return class_to_index;
    }

private:
    static constexpr uint32_t num_train_samples{60000};
    static constexpr uint32_t num_test_samples{10000};
    static constexpr uint32_t image_height{28};
    static constexpr uint32_t image_width{28};
    static constexpr uint32_t image_channels{1};
    static constexpr std::array<double, image_channels> mean{0.1307}; //TODO: used MNIST values
    static constexpr std::array<double, image_channels> std{0.3081}; //TODO: used MNIST values
};



#endif //FASHIONMNISTINFO_H

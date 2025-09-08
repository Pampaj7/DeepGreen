#ifndef FASHIONMNIST_H
#define FASHIONMNIST_H

#include "DatasetInfo.h"


class FashionMNIST final : public DatasetInfo<FashionMNIST> {
    friend class DatasetInfo<FashionMNIST>;

private:
    static constexpr uint32_t num_classes = 10;
    static constexpr uint32_t num_train_samples = 60000;
    static constexpr uint32_t num_test_samples = 10000;
    static constexpr uint32_t image_height = 28;
    static constexpr uint32_t image_width = 28;
    static constexpr uint32_t image_channels = 1;
    static constexpr std::array<double, image_channels> mean{0.1307}; //TODO: used MNIST values
    static constexpr std::array<double, image_channels> std{0.3081}; //TODO: used MNIST values

    static constexpr auto dataset_name = "Fashion-MNIST";
    static constexpr auto train_folder = "train";
    static constexpr auto test_folder = "test";
};



#endif //FASHIONMNIST_H

#ifndef CIFAR100_H
#define CIFAR100_H

#include "DatasetInfo.h"


class CIFAR100 final : public DatasetInfo<CIFAR100> {
    friend class DatasetInfo<CIFAR100>;

private:
    static constexpr uint32_t num_train_samples = 50000;
    static constexpr uint32_t num_test_samples = 10000;
    static constexpr uint32_t image_height = 32;
    static constexpr uint32_t image_width = 32;
    static constexpr uint32_t image_channels = 3;
    static constexpr std::array<double, image_channels> mean{0.4914, 0.4822, 0.4465};
    static constexpr std::array<double, image_channels> std{0.2470, 0.2434, 0.2616};

    static constexpr auto dataset_name = "CIFAR-100";
    static constexpr auto train_folder = "train";
    static constexpr auto test_folder = "test";
    static constexpr auto num_classes = CIFAR100_NUM_CLASSES;
};



#endif //CIFAR100_H

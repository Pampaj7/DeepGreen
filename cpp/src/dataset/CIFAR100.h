#ifndef CIFAR100_H
#define CIFAR100_H

#include "DatasetInfo.h"


class CIFAR100 final : public DatasetInfo<CIFAR100> {
    friend class DatasetInfo<CIFAR100>;

private:
    static constexpr uint32_t num_classes = 100;
    static constexpr uint32_t num_train_samples = 50000;
    static constexpr uint32_t num_test_samples = 10000;
    static constexpr uint32_t image_height = 32;
    static constexpr uint32_t image_width = 32;
    static constexpr uint32_t image_channels = 3;
    static constexpr std::array<double, image_channels> mean{0.5071, 0.4867, 0.4408};
    static constexpr std::array<double, image_channels> std{0.2675, 0.2565, 0.2761};

    static constexpr auto dataset_name = "CIFAR-100";
    static constexpr auto train_folder = "train";
    static constexpr auto test_folder = "test";
};



#endif //CIFAR100_H

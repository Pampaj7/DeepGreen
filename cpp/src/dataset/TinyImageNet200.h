#ifndef TINYIMAGENET200_H
#define TINYIMAGENET200_H

#include "DatasetInfo.h"


class TinyImageNet200 final : public DatasetInfo<TinyImageNet200> {
    friend class DatasetInfo<TinyImageNet200>;

private:
    static constexpr uint32_t num_train_samples = 100000;
    static constexpr uint32_t num_test_samples = 10000;
    static constexpr uint32_t image_height = 64;
    static constexpr uint32_t image_width = 64;
    static constexpr uint32_t image_channels = 3;
    static constexpr std::array<double, image_channels> mean{0.4914, 0.4822, 0.4465}; //TODO
    static constexpr std::array<double, image_channels> std{0.2470, 0.2434, 0.2616}; //TODO

    static constexpr auto dataset_name = "Tiny ImageNet-200";
    static constexpr auto train_folder = "train";
    static constexpr auto test_folder = "val";
    static constexpr auto num_classes = TINY_IMAGENET200_NUM_CLASSES;
};



#endif //TINYIMAGENET200_H

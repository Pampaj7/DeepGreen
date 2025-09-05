#ifndef TINYIMAGENET200_H
#define TINYIMAGENET200_H

#include "DatasetInfo.h"


class TinyImageNet200 final : public DatasetInfo<TinyImageNet200> {
    friend class DatasetInfo<TinyImageNet200>;

private:
    static constexpr uint32_t num_classes = 200;
    static constexpr uint32_t num_train_samples = 100000;
    static constexpr uint32_t num_test_samples = 10000;
    static constexpr uint32_t image_height = 64;
    static constexpr uint32_t image_width = 64;
    static constexpr uint32_t image_channels = 3;
    static constexpr std::array<double, image_channels> mean{0.485,0.456,0.406}; //TODO
    static constexpr std::array<double, image_channels> std{0.229,0.224,0.225}; //TODO

    static constexpr auto dataset_name = "Tiny ImageNet-200";
    static constexpr auto train_folder = "train";
    static constexpr auto test_folder = "val";
};



#endif //TINYIMAGENET200_H

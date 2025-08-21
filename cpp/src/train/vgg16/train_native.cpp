#include "train_native.h"

#include <iostream>

#include "dataset/CIFAR100.h"
#include "train/vgg16/train_vgg16.h"


// Where to find the CIFAR-100 dataset.
const char* kCifarRelativePath = "../data/cifar100_png";
const char* kCifarClassesJson = "classes.json";

// The batch size for training.
constexpr int32_t kTrainBatchSize = 64; //TODO: 128
// The batch size for testing.
constexpr int32_t kTestBatchSize = 128;
// The number of epochs to train.
constexpr int32_t kNumberOfEpochs = 1; //TODO: 30



int main() {
    try {
        train_native<CIFAR100>(kCifarRelativePath, kCifarClassesJson, 32,
            kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);


    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

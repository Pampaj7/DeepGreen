#include <iostream>

#include "dataset/CIFAR100.h"
#include "train/vgg16/train_vgg16.h"


// Where to find the CIFAR-100 dataset.
const char* kCifarRootRelativePath = "../data/cifar100_png";
const char* kCifarClassesJson = "classes.json";

// VGG-16 model for CIFAR-100
const char* kVggCifarFilename = VGG16_CIFAR100_FILENAME;

// The batch size for training.
constexpr int32_t kTrainBatchSize = 128;
// The batch size for testing.
constexpr int32_t kTestBatchSize = 128;
// The number of epochs to train.
constexpr int32_t kNumberOfEpochs = 30;



int main() {
    try {
        train_vgg16<CIFAR100>(kCifarRootRelativePath, kCifarClassesJson, kVggCifarFilename,
            kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
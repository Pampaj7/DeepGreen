#include <iostream>

#include "dataset/CIFAR100.h"
#include "train/resnet18/train_resnet18.h"


// Where to find the CIFAR-100 dataset.
const char* kCifarRelativePath = "../data/cifar100_png";
const char* kCifarClassesJson = "classes.json";

// ResNet-18 model for CIFAR-100
const char* kResnetCifarFilename = RESNET18_CIFAR100_FILENAME;

// The batch size for training.
constexpr int32_t kTrainBatchSize = 64;
// The batch size for testing.
constexpr int32_t kTestBatchSize = 128;
// The number of epochs to train.
constexpr int32_t kNumberOfEpochs = 30;



int main() {
    try {
        train_resnet18<CIFAR100>(kCifarRelativePath, kCifarClassesJson, kResnetCifarFilename,
            kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
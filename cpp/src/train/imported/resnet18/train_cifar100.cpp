#include <iostream>

#include "dataset/CIFAR100.h"
#include "train/imported/resnet18/train_resnet18.h"


// Where to find the CIFAR-100 dataset.
const char* kCifarRelativePath = "../data/cifar100_png";
const char* kCifarClassesJson = "classes.json";

// ResNet-18 model for CIFAR-100
const char* kResnetCifarFilename = RESNET18_CIFAR100_FILENAME;

// The image resize value (single value for both dimensions).
constexpr int32_t imageSize = 32;
// The batch size for training.
constexpr int32_t kTrainBatchSize = 128;
// The batch size for testing.
constexpr int32_t kTestBatchSize = 128;
// The number of epochs to train.
constexpr int32_t kNumberOfEpochs = 30;

// File name in which to save results
const std::string outputFileName = "resnet18_cifar100";



int main() {
    try {
        train_resnet18<CIFAR100>(outputFileName, kCifarRelativePath, kCifarClassesJson,
            kResnetCifarFilename, imageSize, kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
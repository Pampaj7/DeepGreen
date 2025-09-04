#include <iostream>

#include "dataset/CIFAR100.h"
#include "train/native/vgg16/train_vgg16.h"


// Where to find the CIFAR-100 dataset.
const char* kCifarRelativePath = "../data/cifar100_png";
const char* kCifarClassesJson = "classes.json";

// The batch size for training.
constexpr int32_t kTrainBatchSize = 128;
// The batch size for testing.
constexpr int32_t kTestBatchSize = 128;
// The number of epochs to train.
constexpr int32_t kNumberOfEpochs = 1; //TODO: 30

// File name in which to save results
const std::string outputFileName = "vgg16_cifar100";



int main() {
    try {
        train_vgg16<CIFAR100>(outputFileName, kCifarRelativePath, kCifarClassesJson,
            kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);


    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

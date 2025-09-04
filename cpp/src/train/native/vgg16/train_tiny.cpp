#include <iostream>

#include "dataset/TinyImageNet200.h"
#include "train/native/vgg16/train_vgg16.h"


// Where to find the Tiny ImageNet-200 dataset.
const char* kTinyRootRelativePath = "../data/tiny_imagenet_png";
const char* kTinyClassesJson = "classes.json";

// The batch size for training.
constexpr int32_t kTrainBatchSize = 128;
// The batch size for testing.
constexpr int32_t kTestBatchSize = 128;
// The number of epochs to train.
constexpr int32_t kNumberOfEpochs = 30;

// File name in which to save results
const std::string outputFileName = "vgg16_tiny";



int main() {
    try {
        train_vgg16<TinyImageNet200>(outputFileName, kTinyRootRelativePath, kTinyClassesJson,
            kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
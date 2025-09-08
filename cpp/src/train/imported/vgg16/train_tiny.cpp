#include <iostream>

#include "dataset/TinyImageNet200.h"
#include "train/imported/vgg16/train_vgg16.h"


// Where to find the Tiny ImageNet-200 dataset.
const char* kTinyRootRelativePath = "../data/tiny_imagenet_png";
const char* kTinyClassesJson = "classes.json";

// VGG-16 model for Tiny ImageNet-200
const char* kVggTinyFilename = VGG16_TINYIMAGENET200_FILENAME;

// The image resize value (single value for both dimensions).
constexpr int32_t imageSize = 32;
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
            kVggTinyFilename, imageSize, kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
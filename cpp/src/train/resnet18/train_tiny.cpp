#include <iostream>

#include "dataset/TinyImageNet200.h"
#include "train_resnet18.h"


// Where to find the Tiny ImageNet-200 dataset.
const char* kTinyRelativePath = "../data/tiny_imagenet_png";
const char* kTinyClassesJson = "classes.json";

// ResNet-18 model for Tiny ImageNet-200
const char* kResnetTinyFilename = RESNET18_TINYIMAGENET200_FILENAME;

// The batch size for training.
constexpr int64_t kTrainBatchSize = 64;
// The batch size for testing.
constexpr int64_t kTestBatchSize = 1000; //TODO
// The number of epochs to train.
constexpr int64_t kNumberOfEpochs = 1;



int main() {
    try {

        train_resnet18<TinyImageNet200>(kTinyRelativePath, kTinyClassesJson, kResnetTinyFilename,
            kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
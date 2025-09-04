#include <iostream>

#include "dataset/FashionMNIST.h"
#include "train/imported/vgg16/train_vgg16.h"


// Where to find the Fashion-MNIST dataset.
const char* kFashionRootRelativePath = "../data/fashion_mnist_png";
const char* kFashionClassesJson = "classes.json";

// VGG-16 model for Fashion-MNIST
const char* kVggFashionFilename = VGG16_FASHIONMNIST_FILENAME;

// The batch size for training.
constexpr int32_t kTrainBatchSize = 128;
// The batch size for testing.
constexpr int32_t kTestBatchSize = 128;
// The number of epochs to train.
constexpr int32_t kNumberOfEpochs = 30;

// File name in which to save results
const std::string outputFileName = "vgg16_fashion";



int main() {
    try {
        train_vgg16<FashionMNIST>(outputFileName, kFashionRootRelativePath, kFashionClassesJson,
            kVggFashionFilename, kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
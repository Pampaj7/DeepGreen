#include <iostream>

#include "dataset/FashionMNIST.h"
#include "train_vgg16.h"


// Where to find the Fashion-MNIST dataset.
const char* kFashionRootRelativePath = "../data/fashion_mnist_png";
const char* kFashionClassesJson = "classes.json";

// VGG-16 model for Fashion-MNIST
const char* kVggFashionFilename = VGG16_FASHIONMNIST_FILENAME;

// The batch size for training.
constexpr int64_t kTrainBatchSize = 64;
// The batch size for testing.
constexpr int64_t kTestBatchSize = 1000; //TODO
// The number of epochs to train.
constexpr int64_t kNumberOfEpochs = 1;



int main() {
    try {

        train_vgg16<FashionMNIST>(kFashionRootRelativePath, kFashionClassesJson, kVggFashionFilename,
            kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
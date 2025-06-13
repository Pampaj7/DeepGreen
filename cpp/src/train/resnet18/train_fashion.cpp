#include <iostream>

#include "dataset/FashionMNIST.h"
#include "train/resnet18/train_resnet18.h"


// Where to find the Fashion-MNIST dataset.
const char* kFashionRelativePath = "../data/fashion_mnist_png";
const char* kFashionClassesJson = "classes.json";

// ResNet-18 model for Fashion-MNIST
const char* kResnetFashionFilename = RESNET18_FASHIONMNIST_FILENAME;

// The batch size for training.
constexpr int64_t kTrainBatchSize = 64;
// The batch size for testing.
constexpr int64_t kTestBatchSize = 1000; //TODO
// The number of epochs to train.
constexpr int64_t kNumberOfEpochs = 1;



int main() {
    try {

        train_resnet18<FashionMNIST>(kFashionRelativePath, kFashionClassesJson, kResnetFashionFilename,
            kTrainBatchSize, kTestBatchSize, kNumberOfEpochs);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
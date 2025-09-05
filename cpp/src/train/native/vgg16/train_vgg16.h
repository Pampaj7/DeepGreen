#ifndef TRAIN_NATIVE_H
#define TRAIN_NATIVE_H
#include <torch/torch.h>

#include "model/vgg16.h"
#include "train/native/train_model.h"

// Minimum size required by VGG-16 for the feature vector not to cancel out
constexpr int32_t kVggMinImageSize = 32;

template <typename Dataset>
void train_vgg16(const std::string& outputFileName, const char* dataRootRelativePath, const char* classesJson,
    const int32_t trainBatchSize, const int32_t testBatchSize, const int32_t numberOfEpochs)
{
    // create vgg16
    models::VGG16 vgg16(Dataset::getNumClasses(), false);
    /*vgg16->apply(
        [](torch::nn::Module& m) {
            if (auto* conv = m.as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(conv->weight);
            torch::nn::init::zeros_(conv->bias);
            }
            else if (auto* linear = m.as<torch::nn::Linear>()) {
                torch::nn::init::kaiming_normal_(linear->weight);
                torch::nn::init::zeros_(linear->bias);
            }
        }
    );*/

    train_model<models::VGG16, Dataset>(outputFileName, dataRootRelativePath, classesJson,
        vgg16, kVggMinImageSize,
        trainBatchSize, testBatchSize, numberOfEpochs);
}


#endif //TRAIN_NATIVE_H

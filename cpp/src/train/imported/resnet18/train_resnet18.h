#ifndef TRAIN_RESNET18_H
#define TRAIN_RESNET18_H
#include "train/imported/train_model.h"

// Minimum size required by ResNet-18 for the feature vector not to cancel out
constexpr int32_t kResNetMinImageSize = 28;

template <typename Dataset>
void train_resnet18(const std::string& outputFileName, const char* dataRootRelativePath, const char* classesJson,
    const char* resnet_dataset_filename, const int32_t imgResize, const int32_t trainBatchSize, const int32_t testBatchSize,
    const int32_t numberOfEpochs)
{
    if (imgResize < kResNetMinImageSize)
        throw std::invalid_argument(
            "ResNet-18 requires image sizes to be at least " +
            std::to_string(kResNetMinImageSize) + "x" +
            std::to_string(kResNetMinImageSize) + " pixels");

    train_model<Dataset>(outputFileName, dataRootRelativePath, classesJson,
        resnet_dataset_filename, imgResize,
        trainBatchSize, testBatchSize, numberOfEpochs);
}



#endif //TRAIN_RESNET18_H

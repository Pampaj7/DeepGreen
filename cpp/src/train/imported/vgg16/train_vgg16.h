#ifndef TRAIN_VGG16_H
#define TRAIN_VGG16_H
#include "train/imported/train_model.h"

// Minimum size required by VGG-16 for the feature vector not to cancel out
constexpr int32_t kVggMinImageSize = 32;

template <typename Dataset>
void train_vgg16(const std::string& outputFileName, const char* dataRootRelativePath, const char* classesJson,
    const char* vgg_dataset_filename, const int32_t imgResize, const int32_t trainBatchSize, const int32_t testBatchSize,
    const int32_t numberOfEpochs)
{
    if (imgResize < kVggMinImageSize)
        throw std::invalid_argument(
            "VGG-16 requires image sizes to be at least " +
            std::to_string(kVggMinImageSize) + "x" +
            std::to_string(kVggMinImageSize) + " pixels");

    train_model<Dataset>(outputFileName, dataRootRelativePath, classesJson,
        vgg_dataset_filename, imgResize,
        trainBatchSize, testBatchSize, numberOfEpochs);
}



#endif //TRAIN_VGG16_H
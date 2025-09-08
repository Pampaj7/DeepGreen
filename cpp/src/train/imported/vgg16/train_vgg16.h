#ifndef TRAIN_VGG16_H
#define TRAIN_VGG16_H
#include "train/imported/train_model.h"

// Minimum size required by VGG-16 for the feature vector not to cancel out
constexpr int32_t kVggMinImageSize = 32;

template <typename Dataset>
void train_vgg16(const std::string& outputFileName, const char* dataRootRelativePath, const char* classesJson,
    const char* vgg_dataset_filename, const int32_t trainBatchSize, const int32_t testBatchSize,
    const int32_t numberOfEpochs)
{
    train_model<Dataset>(outputFileName, dataRootRelativePath, classesJson,
        vgg_dataset_filename, kVggMinImageSize,
        trainBatchSize, testBatchSize, numberOfEpochs);
}



#endif //TRAIN_VGG16_H
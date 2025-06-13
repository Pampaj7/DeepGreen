#ifndef TRAIN_VGG16_H
#define TRAIN_VGG16_H
#include "train/train_model.h"

// Minimum size required by VGG-16 for the feature vector not to cancel out
constexpr int64_t kVggMinImageSize = 32;

template <typename Dataset>
void train_vgg16(const char* dataRootRelativePath, const char* classesJson, const char* vgg_dataset_filename,
    const int64_t trainBatchSize, const int64_t testBatchSize, const int64_t numberOfEpochs)
{
    train_model<Dataset>(dataRootRelativePath, classesJson,
        vgg_dataset_filename, kVggMinImageSize,
        trainBatchSize, testBatchSize, numberOfEpochs);
}



#endif //TRAIN_VGG16_H
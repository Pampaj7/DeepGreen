#ifndef TRAIN_RESNET18_H
#define TRAIN_RESNET18_H
#include "train/train_model.h"

// Minimum size required by ResNet-18 for the feature vector not to cancel out
constexpr int64_t kVggMinImageSize = 28;

template <typename Dataset>
void train_resnet18(const char* dataRootRelativePath, const char* classesJson, const char* resnet_dataset_filename,
    const int64_t trainBatchSize, const int64_t testBatchSize, const int64_t numberOfEpochs)
{
    train_model<Dataset>(dataRootRelativePath, classesJson,
        resnet_dataset_filename, kVggMinImageSize,
        trainBatchSize, testBatchSize, numberOfEpochs);
}



#endif //TRAIN_RESNET18_H

#include <cstdlib>
#include <iostream>

#include "dataset/Imagenette.h"
#include "train/imported/vgg16/train_vgg16.h"


// Where to find the Imagenette dataset.
const char* kImagenetteRelativePath = "../data/imagenette_png";
// The other five cells read their class labels from a classes.json shipped with
// the images. Imagenette's converter writes no such file, so dataset/Imagenette.h
// derives the ten labels from the class directories themselves and uses this only
// to locate the dataset root, which keeps the call identical to the other cells.
const char* kImagenetteClassesJson = "classes.json";

// VGG-16 model for Imagenette
const char* kVggImagenetteFilename = VGG16_IMAGENETTE_FILENAME;

// The image size (single value for both dimensions). The PNGs are already
// 224x224 on disk -- resized once, offline, by
// dataloader/download_convert_imagenette.py -- so nothing is resampled at run
// time (spec S3) and this value is asserted against the dataset, not applied to
// it: kResizeInLoader is false.
constexpr int32_t imageSize = 224;
constexpr bool kResizeInLoader = false;
// The batch size for training.
constexpr int32_t kTrainBatchSize = 32;
// The batch size for testing.
constexpr int32_t kTestBatchSize = 32;
// The number of epochs to train.
constexpr int32_t kNumberOfEpochs = 30;

// The number of dataset loader threads (spec S3). Two, as in every other cell,
// unless the run contract raises it for this one: at 224x224 the host decodes 49
// times the pixels per image, so the campaign may legitimately want more here,
// and the value it used is recorded per run in the manifest.
static int32_t loaderThreadsFromEnvironment()
{
    if (const char* v = std::getenv("DEEPGREEN_LOADER_THREADS")) {
        try {
            const int parsed = std::stoi(v);
            if (parsed > 0)
                return static_cast<int32_t>(parsed);
        } catch (const std::exception&) {
            // fall through to the campaign default
        }
    }
    return 2;
}
const int32_t LOADER_THREADS = loaderThreadsFromEnvironment();

// File name in which to save results
const std::string outputFileName = "vgg16_imagenette";



int main() {
    try {
        train_vgg16<Imagenette>(outputFileName, kImagenetteRelativePath, kImagenetteClassesJson,
            kVggImagenetteFilename, imageSize, kTrainBatchSize, kTestBatchSize, kNumberOfEpochs,
            kResizeInLoader, LOADER_THREADS);

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

#ifndef IMAGENETTE_H
#define IMAGENETTE_H

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "DatasetInfo.h"


// Imagenette: the accelerator-saturation cell (spec S7).
//
// Ten ImageNet classes at 224x224x3, resized once offline by
// dataloader/download_convert_imagenette.py (shorter side to 224, centre crop),
// so nothing is resized at run time -- see train/imported/train_model.h, whose
// resizeInLoader argument this cell passes as false.
//
// Sample counts are the ones on disk, counted after the conversion finished
// (`find data/imagenette_png/{train,test} -name '*.png' | wc -l`): 9,469 and
// 3,925. LazyImageFolder asserts against them, which is a no-op in a Release
// build -- a limitation of the existing loader, not of this cell.
namespace ImagenetteDetail {

    // Aliases, because the DatasetInfo subclasses below declare a static member
    // named `std` (the per-channel standard deviation). Inside the scope of such
    // a class, unqualified `std` names that member and not the namespace, so
    // every type a member of Imagenette mentions is spelled through these.
    using ClassIndexMap = std::map<std::string, int>;
    using PathString = std::string;

    /// The directory of one split, given where that dataset's classes.json
    /// would live (the argument LazyImageFolder passes down).
    inline PathString splitDirectory(const PathString& classes_json_path,
                                     const PathString& split_folder)
    {
        return (std::filesystem::path(classes_json_path).parent_path() / split_folder).string();
    }

    /// Label the class directories under `split_dir` in ascending byte order of
    /// their names. The converter writes them all in lower case, so byte order
    /// is alphabetical order and the labels are
    ///
    ///     0 cassette player   3 english springer   6 gas pump    9 tench
    ///     1 chain saw         4 french horn        7 golf ball
    ///     2 church            5 garbage truck      8 parachute
    ///
    /// which every stack in this cell agrees on. The list is printed once per
    /// run so that the labelling is in the run's own log rather than asserted
    /// in a comment.
    ///
    /// The other five C++ cells read their labels out of a classes.json shipped
    /// beside the images; Imagenette's converter writes no such file, so the
    /// classes come from the directory layout itself, which is what every other
    /// stack in this cell does (torchvision's ImageFolder sorts the entries of
    /// os.listdir()). std::filesystem::directory_iterator returns entries in an
    /// unspecified order, so the sort is what makes the labelling deterministic;
    /// the names are compared and stored whole, so the spaces in "english
    /// springer" and "cassette player" are not special.
    inline ClassIndexMap scanClassDirectories(const PathString& split_dir,
                                              const uint32_t expected_classes)
    {
        if (!std::filesystem::is_directory(split_dir))
            throw std::runtime_error("Imagenette: no such split directory: " + split_dir);

        std::vector<PathString> class_names;
        for (const auto& entry : std::filesystem::directory_iterator(split_dir))
            if (entry.is_directory())
                class_names.push_back(entry.path().filename().string());

        std::sort(class_names.begin(), class_names.end());

        ClassIndexMap class_to_index;
        int idx_label = 0;
        for (const auto& class_name : class_names)
            class_to_index[class_name] = idx_label++;

        if (class_to_index.size() != expected_classes)
            throw std::runtime_error(
                "Imagenette: expected " + std::to_string(expected_classes) +
                " class directories under " + split_dir + ", found " +
                std::to_string(class_to_index.size()) +
                ". The dataset conversion may be incomplete.");

        std::cout << "Imagenette classes (label = directory name, byte order):";
        for (const auto& [class_name, label] : class_to_index)
            std::cout << "\n  " << label << " " << class_name;
        std::cout << std::endl;

        return class_to_index;
    }

}


class Imagenette final : public DatasetInfo<Imagenette> {
    friend class DatasetInfo<Imagenette>;

public:
    /// Hides DatasetInfo::loadClassesToIndexMap: this dataset has no
    /// classes.json, so the ten labels are derived from the training split's
    /// directory layout. `classes_json_path` is where that file would live and
    /// is used only to locate the dataset root, so the call site in
    /// LazyImageFolder stays the one every other cell uses.
    static const ImagenetteDetail::ClassIndexMap& loadClassesToIndexMap(
        const ImagenetteDetail::PathString& classes_json_path);

private:
    static constexpr uint32_t num_classes = 10;
    static constexpr uint32_t num_train_samples = 9469;
    static constexpr uint32_t num_test_samples = 3925;
    static constexpr uint32_t image_height = 224;
    static constexpr uint32_t image_width = 224;
    static constexpr uint32_t image_channels = 3;
    // Declared as every other dataset declares them, and unused for the same
    // reason: S3 fixes the input scaling at [0,1], so the Normalize transform
    // is not composed in train_model.h and these values never reach a tensor.
    static constexpr std::array<double, image_channels> mean{0.485, 0.456, 0.406}; // ImageNet mean
    static constexpr std::array<double, image_channels> std{0.229, 0.224, 0.225}; // ImageNet std

    static constexpr auto dataset_name = "Imagenette";
    static constexpr auto train_folder = "train";
    static constexpr auto test_folder = "test";
};


inline const ImagenetteDetail::ClassIndexMap& Imagenette::loadClassesToIndexMap(
    const ImagenetteDetail::PathString& classes_json_path)
{
    // Function-local static: built once, on first use, thread-safely -- the same
    // contract as the std::call_once in DatasetInfo, which the two splits of one
    // run rely on to share a single labelling.
    static const ImagenetteDetail::ClassIndexMap class_to_index =
        ImagenetteDetail::scanClassDirectories(
            ImagenetteDetail::splitDirectory(classes_json_path, train_folder),
            num_classes);
    return class_to_index;
}



#endif //IMAGENETTE_H

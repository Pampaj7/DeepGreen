#ifndef IMAGEFOLDER_H
#define IMAGEFOLDER_H

// Single entry point for the image-folder dataset used by every C++ target.
//
// The first campaign had two independent implementations and picked between
// them per build target: five of the six *_imported targets listed
// LazyImageFolder.h, resnet18_fashion_imported listed InMemoryImageFolder.h,
// and every native target used LazyImageFolder directly. src/train/imported/
// train_model.h then included this header, which did not exist, so none of the
// imported targets could be built at all -- they are commented out in
// CMakeLists.txt.
//
// Loading strategy is a first-order confound in this study: the audit shows the
// GPU runs at 24-56% of its power limit and that epoch duration accounts for
// essentially the whole energy spread, so how images reach the device dominates
// the measurement. It must therefore be one choice, made explicitly, and the
// same for every target.
//
// Default: lazy loading, matching PyTorch's ImageFolder + DataLoader, which is
// what seven of the eight ecosystems do. Define DEEPGREEN_INMEMORY_DATASET to
// build the preloading variant instead, for a deliberate comparison.

#ifdef DEEPGREEN_INMEMORY_DATASET
#include "InMemoryImageFolder.h"
template <typename Dataset>
using ImageFolder = InMemoryImageFolder<Dataset>;
#else
#include "LazyImageFolder.h"
template <typename Dataset>
using ImageFolder = LazyImageFolder<Dataset>;
#endif

#endif // IMAGEFOLDER_H

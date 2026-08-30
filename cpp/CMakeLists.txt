cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(DeepGreen_cpp)

# Configure OpenCV
if (WIN32)
    # set(OpenCV_DIR "C:/Program Files (x86)/OpenCV/opencv_4.10.0/build")
    # or add "path/to/openCV/build/bin" to PATH env var
elseif (LINUX)
    # Was hard-coded to /home/marcopaglio/tools/opencv-4.11.0/build, so the
    # package built on one machine only. Override with -DOpenCV_DIR=... or the
    # OpenCV_DIR environment variable; otherwise fall back to system discovery.
    if (NOT DEFINED OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
        set(OpenCV_DIR "$ENV{OpenCV_DIR}")
    endif ()
endif ()

find_package(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV found.")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
MESSAGE(STATUS "Python3 found.")
include_directories(${Python3_INCLUDE_DIRS})
set(PY_SCRIPT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/py_script")


option(CUDA_ENABLED "Set CUDA for LibTorch" ON)
if(CUDA_ENABLED) # Using GPU
    set(CMAKE_CUDA_STANDARD 17)
    # Was hard-coded to 89 (L40S). Building for the wrong architecture either
    # fails or silently falls back to JIT compilation at first kernel launch,
    # which would be attributed to the ecosystem in an energy measurement.
    # Override with -DCMAKE_CUDA_ARCHITECTURES=..., default is native detection.
    #   89 L40S / Ada,  86 RTX 30xx,  75 RTX 20xx,  61 GTX 10xx
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif ()

    if (WIN32)
        # Configure CUDA
        set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe")

        # Configure LibTorch
        if (CMAKE_BUILD_TYPE MATCHES "Debug")
            set(LIBTORCH_SPEC "libtorch_win32_cuda_debug")
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.0%2Bcu128.zip")
        elseif (CMAKE_BUILD_TYPE MATCHES "Release")
            set(LIBTORCH_SPEC "libtorch_win32_cuda_release")
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.0%2Bcu128.zip")
        endif ()
    elseif (LINUX)
        # Configure CUDA
        # Was hard-coded to /home/marcopaglio/tools/cuda-12.8. Take it from the
        # environment (CUDA_HOME or CUDAToolkit_ROOT) and only pin nvcc if that
        # actually resolves, so a normal system install works untouched.
        if (NOT DEFINED CUDAToolkit_ROOT)
            if (DEFINED ENV{CUDAToolkit_ROOT})
                set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}")
            elseif (DEFINED ENV{CUDA_HOME})
                set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}")
            endif ()
        endif ()
        if (DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
        endif ()

        # Configure LibTorch
        set(LIBTORCH_SPEC "libtorch_linux_cuda")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip")
    endif ()

    find_package(CUDAToolkit REQUIRED)
    # Add path to nvxt3 (cuda's profiler) which Libtorch (v.2.7.0) cannot be found.
    # Important: insert before "find_package(Torch REQUIRED)"
    if (DEFINED CUDAToolkit_ROOT)
        set(NVTX3_INCLUDE_DIR "${CUDAToolkit_ROOT}/include/nvtx3")
    else ()
        set(NVTX3_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}/nvtx3")
    endif ()
    include_directories(${NVTX3_INCLUDE_DIR})
    enable_language(CUDA)
else () # Using CPU
    # Configure LibTorch
    if (WIN32)
        if (CMAKE_BUILD_TYPE MATCHES "Debug")
            set(LIBTORCH_SPEC "libtorch_win32_cpu_debug")
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.7.0%2Bcpu.zip")
        elseif (CMAKE_BUILD_TYPE MATCHES "Release") #TODO: not tested
            set(LIBTORCH_SPEC "libtorch_win32_cpu_release")
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.0%2Bcpu.zip")
        endif ()
    elseif (LINUX)
        set(LIBTORCH_SPEC "libtorch_linux_cpu")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip")
    endif ()
endif ()

### FETCH CONTENT ###
include(FetchContent)
Set(FETCHCONTENT_QUIET FALSE)

# Download nlohmann JSON Reader
# TODO:Il fetch usando più modalità (Debug, Release, etc) genera un errore nel load del cmake della modalità più lenta, a meno che non fosse già stato salvato
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/tools/json_reader")
FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        # the following option is necessary when cmake version is 3.23 or lower
        # it defines the default behavior of cmake >= 3.24, as recommended at
        # https://cmake.org/cmake/help/latest/policy/CMP0135.html
        DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_MakeAvailable(json)

list(APPEND CMAKE_PREFIX_PATH "${FETCHCONTENT_BASE_DIR}/json-build")
find_package(nlohmann_json REQUIRED)
MESSAGE(STATUS "Tool 'nlohmann_json' found.")

# Download LibTorch
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/tools/libtorch")
FetchContent_Declare(
        ${LIBTORCH_SPEC}
        URL ${LIBTORCH_URL}
        # the following option is necessary when cmake version is 3.23 or lower
        # it defines the default behavior of cmake >= 3.24, as recommended at
        # https://cmake.org/cmake/help/latest/policy/CMP0135.html
        DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_MakeAvailable(${LIBTORCH_SPEC})

list(APPEND CMAKE_PREFIX_PATH "${FETCHCONTENT_BASE_DIR}/${LIBTORCH_SPEC}-src")

# LibTorch's Caffe2Config references CUDA::nvToolsExt, but NVTX left the CUDA
# toolkit in CUDA 12 and is header-only (nvtx3) from then on, so the target does
# not exist and find_package(Torch) fails outright. The submitted package had a
# workaround for this, but gated on WIN32 and placed AFTER find_package(Torch) --
# after the point where it is needed -- so the CUDA build could not configure on
# Linux at all. Define it here, for every platform, before Torch is looked up.
if (CUDA_ENABLED AND NOT TARGET CUDA::nvToolsExt)
    find_library(NVTOOLSEXT_LIBRARY
            NAMES nvToolsExt libnvToolsExt.so.1
            HINTS ${CUDAToolkit_ROOT}/lib ${CUDAToolkit_ROOT}/lib64
                  ${CUDAToolkit_LIBRARY_DIR})
    if (NVTOOLSEXT_LIBRARY)
        message(STATUS "CUDA::nvToolsExt -> ${NVTOOLSEXT_LIBRARY}")
        add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
        set_target_properties(CUDA::nvToolsExt PROPERTIES
                IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}")
    else ()
        message(STATUS "CUDA::nvToolsExt not found: using the header-only NVTX3 shim.")
        add_library(nvtx3_dummy INTERFACE)
        target_include_directories(nvtx3_dummy INTERFACE "${NVTX3_INCLUDE_DIR}")
        add_library(CUDA::nvToolsExt ALIAS nvtx3_dummy)
    endif ()
endif ()

find_package(Torch REQUIRED)
MESSAGE(STATUS "Tool 'LibTorch' found.")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Training with native models
add_executable(resnet18_fashion
        src/train/native/resnet18/train_fashion.cpp
        src/train/native/resnet18/train_resnet18.h
        src/train/native/train_model.h
        src/model/resnet18.cpp
        src/model/resnet18.h
        src/dataset/LazyImageFolder.h
        src/dataset/FashionMNIST.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(resnet18_cifar100
        src/train/native/resnet18/train_cifar100.cpp
        src/train/native/resnet18/train_resnet18.h
        src/train/native/train_model.h
        src/model/resnet18.cpp
        src/model/resnet18.h
        src/dataset/LazyImageFolder.h
        src/dataset/CIFAR100.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(resnet18_tiny
        src/train/native/resnet18/train_tiny.cpp
        src/train/native/resnet18/train_resnet18.h
        src/train/native/train_model.h
        src/model/resnet18.cpp
        src/model/resnet18.h
        src/dataset/LazyImageFolder.h
        src/dataset/TinyImageNet200.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(vgg16_fashion
        src/train/native/vgg16/train_fashion.cpp
        src/train/native/vgg16/train_vgg16.h
        src/train/native/train_model.h
        src/model/vgg16.cpp
        src/model/vgg16.h
        src/dataset/LazyImageFolder.h
        src/dataset/FashionMNIST.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(vgg16_cifar100
        src/train/native/vgg16/train_cifar100.cpp
        src/train/native/vgg16/train_vgg16.h
        src/train/native/train_model.h
        src/model/vgg16.cpp
        src/model/vgg16.h
        src/dataset/LazyImageFolder.h
        src/dataset/CIFAR100.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(vgg16_tiny
        src/train/native/vgg16/train_tiny.cpp
        src/train/native/vgg16/train_vgg16.h
        src/train/native/train_model.h
        src/model/vgg16.cpp
        src/model/vgg16.h
        src/dataset/LazyImageFolder.h
        src/dataset/TinyImageNet200.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)

# Training with TorchScript models: NO MORE USED
# The *_imported targets build the TorchScript models exported from Python.
# They were commented out in the submitted package because
# src/train/imported/train_model.h includes dataset/ImageFolder.h, which did
# not exist. That header now exists and selects one loading strategy for every
# target (lazy by default). Re-enabled here; verify the build before use.
add_executable(resnet18_fashion_imported
        src/train/imported/resnet18/train_fashion.cpp
        src/train/imported/resnet18/train_resnet18.h
        src/train/imported/train_model.h
        src/dataset/ImageFolder.h
        src/dataset/FashionMNIST.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(resnet18_cifar100_imported
        src/train/imported/resnet18/train_cifar100.cpp
        src/train/imported/resnet18/train_resnet18.h
        src/train/imported/train_model.h
        src/dataset/ImageFolder.h
        src/dataset/CIFAR100.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(resnet18_tiny_imported
        src/train/imported/resnet18/train_tiny.cpp
        src/train/imported/resnet18/train_resnet18.h
        src/train/imported/train_model.h
        src/dataset/ImageFolder.h
        src/dataset/TinyImageNet200.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(vgg16_fashion_imported
        src/train/imported/vgg16/train_fashion.cpp
        src/train/imported/vgg16/train_vgg16.h
        src/train/imported/train_model.h
        src/dataset/ImageFolder.h
        src/dataset/FashionMNIST.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(vgg16_cifar100_imported
        src/train/imported/vgg16/train_cifar100.cpp
        src/train/imported/vgg16/train_vgg16.h
        src/train/imported/train_model.h
        src/dataset/ImageFolder.h
        src/dataset/CIFAR100.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)
add_executable(vgg16_tiny_imported
        src/train/imported/vgg16/train_tiny.cpp
        src/train/imported/vgg16/train_vgg16.h
        src/train/imported/train_model.h
        src/dataset/ImageFolder.h
        src/dataset/TinyImageNet200.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/dataset_transforms.cpp
        src/dataset_transforms.h
        src/utils.cpp
        src/utils.h
        src/python/PythonTracker.cpp
        src/python/PythonTracker.h
)

add_executable(training_rest src/time_to_sleep.cpp)

set(TARGETS
        resnet18_fashion
        resnet18_cifar100
        resnet18_tiny
        vgg16_fashion
        vgg16_cifar100
        vgg16_tiny
        resnet18_fashion_imported
        resnet18_cifar100_imported
        resnet18_tiny_imported
        vgg16_fashion_imported
        vgg16_cifar100_imported
        vgg16_tiny_imported
)

foreach (target ${TARGETS})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)

    target_link_libraries(${target}
            ${OpenCV_LIBS}
            ${Python3_LIBRARIES}
    )

    target_link_libraries(${target} nlohmann_json::nlohmann_json)

    target_link_libraries(${target} "${TORCH_LIBRARIES}")
    # The following code block is suggested to be used on Windows.
    # According to https://github.com/pytorch/pytorch/issues/25457,
    # the DLLs need to be copied to avoid memory errors.
    if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET ${target}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:${target}>)
    endif (MSVC)

    if(CUDA_ENABLED)
        set_target_properties(${target} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
        )
    endif ()

    # Copy Python script in build folder
    add_custom_command(
            TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${PY_SCRIPT_PATH}/tracker/tracker_control.py
            ${CMAKE_BINARY_DIR}/tracker_control.py
    )
    if (WIN32)
        add_custom_command(
                TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${PY_SCRIPT_PATH}/tracker/win_patch_codecarbon.py
                ${CMAKE_BINARY_DIR}/win_patch_codecarbon.py
        )
    endif ()
endforeach ()


add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_compile_definitions(CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}")
# Windows requires to know where Python is located
if (WIN32)
    get_filename_component(PYTHON_DIR "${Python3_EXECUTABLE}" DIRECTORY)
    add_compile_definitions(PYTHON_HOME=L"${PYTHON_DIR}")
endif ()


### PYTHON SCRIPTS ###
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_script")
include(RunPythonWithAutoInstall)

# Check if codecarbon is importable, or install it
include(InstallCodecarbon)
install_codecarbon()

# Download and convert datasets to png-datasets
include(DownloadAndConvertDataset)

# Fashion-MNIST
set(FASHION_MNIST_NAME "Fashion-MNIST")
option(DOWNLOAD_AND_CONVERT_FASHIONMNIST "Download the Fashion-MNIST dataset from the internet and convert to PNGs" OFF)
if (DOWNLOAD_AND_CONVERT_FASHIONMNIST)
    download_and_convert_dataset(${FASHION_MNIST_NAME} "fashion_mnist_png" "download_convert_fashion")
endif()

# Cifar100
set(CIFAR100_NAME "CIFAR-100")
option(DOWNLOAD_AND_CONVERT_CIFAR100 "Download the CIFAR-100 dataset from the internet and convert to PNGs" OFF)
if (DOWNLOAD_AND_CONVERT_CIFAR100)
    download_and_convert_dataset(${CIFAR100_NAME} "cifar100_png" "download_convert_cifar100")
endif()

# Tiny ImageNet-200
set(TINY_IMAGENET200_NAME "Tiny ImageNet-200")
option(DOWNLOAD_AND_CONVERT_TINYIMAGENET200 "Download the Tiny ImageNet-200 dataset from the internet and convert to PNGs" OFF)
if (DOWNLOAD_AND_CONVERT_TINYIMAGENET200)
    download_and_convert_dataset(${TINY_IMAGENET200_NAME} "tiny_imagenet_png" "download_convert_tinyimage")
endif()

# The *_imported targets train the SAME module every other torch-based stack
# trains, copied in from models/ -- one seeded export, made by
# scripts/export_torchscript_models.py, with models/MANIFEST.json recording the
# parameter count and a hash of the weights.
#
# It said "exported once" and was exported twice. This build ran its own copy of
# the export script, unseeded, into CMAKE_BINARY_DIR, so C++ and Rust trained
# networks from different random draws while the manuscript called them a
# byte-identical module -- and the collapse analysis rested on exactly that.
# The submitted version was worse still: this was commented out and C++ trained
# a hand-written port.
# Generate models for each dataset
include(GenerateModel)
set(FASHION_MNIST_NUM_CLASSES 10)
add_compile_definitions(FASHION_MNIST_NUM_CLASSES="${FASHION_MNIST_NUM_CLASSES}")
set(CIFAR100_NUM_CLASSES 100)
add_compile_definitions(CIFAR100_NUM_CLASSES="${CIFAR100_NUM_CLASSES}")
set(TINY_IMAGENET200_NUM_CLASSES 200)
add_compile_definitions(TINY_IMAGENET200_NUM_CLASSES="${TINY_IMAGENET200_NUM_CLASSES}")

# ResNet-18
set(RESNET18_NAME "ResNet-18")
export_model_for_dataset(${RESNET18_NAME} ${FASHION_MNIST_NAME} ${FASHION_MNIST_NUM_CLASSES})
export_model_for_dataset(${RESNET18_NAME} ${CIFAR100_NAME} ${CIFAR100_NUM_CLASSES})
export_model_for_dataset(${RESNET18_NAME} ${TINY_IMAGENET200_NAME} ${TINY_IMAGENET200_NUM_CLASSES})

# VGG-16
set(VGG16_NAME "VGG-16")
export_model_for_dataset(${VGG16_NAME} ${FASHION_MNIST_NAME} ${FASHION_MNIST_NUM_CLASSES})
export_model_for_dataset(${VGG16_NAME} ${CIFAR100_NAME} ${CIFAR100_NUM_CLASSES})
export_model_for_dataset(${VGG16_NAME} ${TINY_IMAGENET200_NAME} ${TINY_IMAGENET200_NUM_CLASSES})


add_custom_target(run_all
    COMMAND resnet18_cifar100
    COMMAND training_rest
    COMMAND resnet18_fashion
    COMMAND training_rest
    COMMAND resnet18_tiny
    COMMAND training_rest
    COMMAND vgg16_cifar100
    COMMAND training_rest
    COMMAND vgg16_fashion
    COMMAND training_rest
    COMMAND vgg16_tiny
    DEPENDS resnet18_cifar100 resnet18_fashion resnet18_tiny vgg16_cifar100 vgg16_fashion vgg16_tiny training_rest
    USES_TERMINAL
)
# Run all with:
# $ cmake --build build --target run_all --config Release
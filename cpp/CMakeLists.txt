cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(DeepGreen_cpp)

# Configure OpenCV
if (WIN32)
    # set(OpenCV_DIR "C:/Program Files (x86)/OpenCV/opencv_4.10.0/build")
    # or add "path/to/openCV/build/bin" to PATH env var
elseif (LINUX)
    set(OpenCV_DIR "/home/marcopaglio/tools/opencv-4.11.0/build")
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
    set(CMAKE_CUDA_ARCHITECTURES 89) # Based on your GPU:
    # 89 for L40S
    # 86 for RTX 30xx
    # 75 for RTX 20xx
    # 61 for GTX 10xx

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
        set(CUDAToolkit_ROOT  "/home/marcopaglio/tools/cuda-12.8")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc") # only needed when run cmake remotely

        # Configure LibTorch
        set(LIBTORCH_SPEC "libtorch_linux_cuda")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip")
    endif ()

    find_package(CUDAToolkit REQUIRED)
    # Add path to nvxt3 (cuda's profiler) which Libtorch (v.2.7.0) cannot be found.
    # Important: insert before "find_package(Torch REQUIRED)"
    set(NVTX3_INCLUDE_DIR "${CUDAToolkit_ROOT}/include/nvtx3")
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
find_package(Torch REQUIRED)
MESSAGE(STATUS "Tool 'LibTorch' found.")


# Insert after "find_package(Torch REQUIRED)"
if (CUDA_ENABLED AND WIN32)
    # Workaround to simulate CUDA::nvToolsExt with NVTX3 (header-only)
    if(NOT TARGET CUDA::nvToolsExt)
        message(STATUS "CUDA::nvToolsExt not found: creating a NVTX3 header-only fake target.")
        add_library(nvtx3_dummy INTERFACE)
        target_include_directories(nvtx3_dummy INTERFACE "${NVTX3_INCLUDE_DIR}")
        add_library(CUDA::nvToolsExt ALIAS nvtx3_dummy)
    endif()
endif ()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(resnet18_fashion
        src/train/resnet18/train_fashion.cpp
        src/train/resnet18/train_resnet18.h
        src/train/train_model.h
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
add_executable(resnet18_cifar100
        src/train/resnet18/train_cifar100.cpp
        src/train/resnet18/train_resnet18.h
        src/train/train_model.h
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
add_executable(resnet18_tiny
        src/train/resnet18/train_tiny.cpp
        src/train/resnet18/train_resnet18.h
        src/train/train_model.h
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
add_executable(vgg16_fashion
        src/train/vgg16/train_fashion.cpp
        src/train/vgg16/train_vgg16.h
        src/train/train_model.h
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
add_executable(vgg16_cifar100
        src/train/vgg16/train_cifar100.cpp
        src/train/vgg16/train_vgg16.h
        src/train/train_model.h
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
add_executable(vgg16_tiny
        src/train/vgg16/train_tiny.cpp
        src/train/vgg16/train_vgg16.h
        src/train/train_model.h
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
set(TARGETS
        resnet18_fashion
        resnet18_cifar100
        resnet18_tiny
        vgg16_fashion
        vgg16_cifar100
        vgg16_tiny
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
include(GenerateModel)
include(DownloadAndConvertDataset)
include(InstallCodecarbon)

# Check if codecarbon is importable, or install it
install_codecarbon()

# Generate ResNet-18 and VGG-16 models for each dataset
set(RESNET18_NAME "ResNet-18")
set(VGG16_NAME "VGG-16")

# Fashion-MNIST
set(FASHION_MNIST_NAME "Fashion-MNIST")
set(FASHION_MNIST_NUM_CLASSES 10)
export_model_for_dataset(${RESNET18_NAME} ${FASHION_MNIST_NAME} ${FASHION_MNIST_NUM_CLASSES})
export_model_for_dataset(${VGG16_NAME} ${FASHION_MNIST_NAME} ${FASHION_MNIST_NUM_CLASSES})
add_compile_definitions(FASHION_MNIST_NUM_CLASSES="${FASHION_MNIST_NUM_CLASSES}")

option(DOWNLOAD_AND_CONVERT_FASHIONMNIST "Download the Fashion-MNIST dataset from the internet and convert to PNGs" OFF)
if (DOWNLOAD_AND_CONVERT_FASHIONMNIST)
    download_and_convert_dataset(${FASHION_MNIST_NAME} "fashion_mnist_png" "download_convert_fashion")
endif()

# Cifar100
set(CIFAR100_NAME "CIFAR-100")
set(CIFAR100_NUM_CLASSES 100)
export_model_for_dataset(${RESNET18_NAME} ${CIFAR100_NAME} ${CIFAR100_NUM_CLASSES})
export_model_for_dataset(${VGG16_NAME} ${CIFAR100_NAME} ${CIFAR100_NUM_CLASSES})
add_compile_definitions(CIFAR100_NUM_CLASSES="${CIFAR100_NUM_CLASSES}")

option(DOWNLOAD_AND_CONVERT_CIFAR100 "Download the CIFAR-100 dataset from the internet and convert to PNGs" OFF)
if (DOWNLOAD_AND_CONVERT_CIFAR100)
    download_and_convert_dataset(${CIFAR100_NAME} "cifar100_png" "download_convert_cifar100")
endif()

# Tiny ImageNet-200
set(TINY_IMAGENET200_NAME "Tiny ImageNet-200")
set(TINY_IMAGENET200_NUM_CLASSES 200)
export_model_for_dataset(${RESNET18_NAME} ${TINY_IMAGENET200_NAME} ${TINY_IMAGENET200_NUM_CLASSES})
export_model_for_dataset(${VGG16_NAME} ${TINY_IMAGENET200_NAME} ${TINY_IMAGENET200_NUM_CLASSES})
add_compile_definitions(TINY_IMAGENET200_NUM_CLASSES="${TINY_IMAGENET200_NUM_CLASSES}")

option(DOWNLOAD_AND_CONVERT_TINYIMAGENET200 "Download the Tiny ImageNet-200 dataset from the internet and convert to PNGs" OFF)
if (DOWNLOAD_AND_CONVERT_TINYIMAGENET200)
    download_and_convert_dataset(${TINY_IMAGENET200_NAME} "tiny_imagenet_png" "download_convert_tinyimage")
endif()
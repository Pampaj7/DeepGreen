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
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")

        # Configure LibTorch
        set(LIBTORCH_SPEC "libtorch_win32_cuda")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.0%2Bcu128.zip")
    elseif (LINUX)
        # Configure CUDA
        set(CUDA_TOOLKIT_ROOT_DIR "/home/marcopaglio/tools/cuda-12.8")
        set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc") # only needed when run cmake remotely

        # Configure LibTorch
        set(LIBTORCH_SPEC "libtorch_linux_cuda")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip")
    endif ()

    # Add path to nvxt3 (cuda's profiler) which Libtorch (v.2.7.0) cannot be found.
    # Important: insert before "find_package(Torch REQUIRED)"
    set(NVTX3_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include/nvtx3")
    include_directories(${NVTX3_INCLUDE_DIR})
    enable_language(CUDA)
else () # Using CPU
    # Configure LibTorch
    if (WIN32)
        set(LIBTORCH_SPEC "libtorch_win32_cpu")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.7.0%2Bcpu.zip")
    elseif (LINUX)
        set(LIBTORCH_SPEC "libtorch_linux_cpu")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip")
    endif ()
endif ()

### FETCH CONTENT ###
include(FetchContent)
Set(FETCHCONTENT_QUIET FALSE)

# Download nlohmann JSON Reader
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


add_executable(resnet18_cifar100
        src/train_cifar100.cpp
        src/dataset/CIFAR100.cpp
        src/dataset/CIFAR100.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/utils.cpp
        src/utils.h
)
add_executable(resnet18_tiny
        src/train_tiny.cpp
        src/dataset/TinyImageNet200.cpp
        src/dataset/TinyImageNet200.h
        src/cnn_function.h
        src/cnn_setup.cpp
        src/cnn_setup.h
        src/utils.cpp
        src/utils.h
)
set(TARGETS
        resnet18_cifar100
        resnet18_tiny
)

foreach (target ${TARGETS})
    target_link_libraries(${target} ${OpenCV_LIBS})

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
endforeach ()


add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_compile_definitions(CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}")



### PYTHON SCRIPTS ###
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_script")
include(RunPythonWithAutoInstall)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Generate ResNet-18 model for each dataset
set(PY_SCRIPT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/py_script")

# Cifar100
set(CIFAR_OUTPUT_FILENAME "resnet18_cifar100")
set(CIFAR_NUM_CLASSES 100)
message(STATUS "Exporting ResNet-18 model to train on CIFAR-100 dataset...")
run_python_script_with_auto_install(
        SCRIPT "${PY_SCRIPT_PATH}/resnet18.py" #TODO: usare modello pytorch
        ARGS ${CIFAR_OUTPUT_FILENAME} ${CIFAR_NUM_CLASSES}
        RESULT_VARIABLE resnet_cifar_success
        ERROR_VARIABLE resnet_cifar_error
)
if(resnet_cifar_success)
    message(STATUS "ResNet-18 for CIFAR-100 successfully exported.")
    add_compile_definitions(CIFAR100_FILENAME="${CIFAR_OUTPUT_FILENAME}.pt")
    add_compile_definitions(CIFAR100_NUM_CLASSES="${CIFAR_NUM_CLASSES}")
else()
    message(FATAL_ERROR "Error exporting ResNet-18 model for CIFAR-100:\n${resnet_cifar_error}")
endif()

# Tiny ImageNet-200
set(TINY_OUTPUT_FILENAME "resnet18_tiny")
set(TINY_IMAGENET_NUM_CLASSES 200)
message(STATUS "Exporting ResNet-18 model to train on Tiny ImageNet-200 dataset...")
run_python_script_with_auto_install(
        SCRIPT "${PY_SCRIPT_PATH}/resnet18.py" #TODO: usare modello pytorch
        ARGS ${TINY_OUTPUT_FILENAME} ${TINY_IMAGENET_NUM_CLASSES}
        RESULT_VARIABLE resnet_tiny_success
        ERROR_VARIABLE resnet_tiny_error
)
if(resnet_tiny_success)
    message(STATUS "ResNet-18 for Tiny ImageNet-200 successfully exported.")
    add_compile_definitions(TINY_IMAGENET200_FILENAME="${TINY_OUTPUT_FILENAME}.pt")
    add_compile_definitions(TINY_IMAGENET200_NUM_CLASSES="${TINY_IMAGENET_NUM_CLASSES}")
else()
    message(FATAL_ERROR "Error exporting ResNet-18 model for Tiny ImageNet-200:\n${resnet_tiny_error}")
endif()



option(DOWNLOAD_AND_CONVERT_CIFAR100 "Download the CIFAR-100 dataset from the internet and convert to PNGs" OFF)
if (DOWNLOAD_AND_CONVERT_CIFAR100)
    set(OUTPUT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../data/cifar100_png")
    #set(DATASET_ROOT "${CMAKE_CURRENT_LIST_DIR}/../data")

    message(STATUS "Downloading and converting CIFAR100 dataset...")
    run_python_script_with_auto_install(
            SCRIPT "${CMAKE_CURRENT_LIST_DIR}/../dataloader/download_convert_cifar100.py"
            ARGS ${OUTPUT_ROOT} #${DATASET_ROOT}
            RESULT_VARIABLE cifar_success
            ERROR_VARIABLE cifar_error
    )
    if(cifar_success)
        message(STATUS "CIFAR-100 successfully downloaded and converted.")
    else()
        message(FATAL_ERROR "Error downloading or converting CIFAR-100:\n${cifar_error}")
    endif()
endif()


option(DOWNLOAD_AND_CONVERT_TINYIMAGENET200 "Download the Tiny ImageNet-200 dataset from the internet and convert to PNGs" OFF)
if (DOWNLOAD_AND_CONVERT_TINYIMAGENET200)
    set(OUTPUT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../data/tiny_imagenet_png")
    #set(DATASET_ROOT "${CMAKE_CURRENT_LIST_DIR}/../data")

    if (WIN32)
        message(WARNING "For Windows users, the following script may fail due to using too long directory paths during conversion.\nIn this case, please, enable long paths in Windows.")
    endif ()
    message(STATUS "Downloading and converting Tiny ImageNet-200 dataset...")
    run_python_script_with_auto_install(
            SCRIPT "${CMAKE_CURRENT_LIST_DIR}/../dataloader/download_convert_tinyimage.py"
            ARGS ${OUTPUT_ROOT} #${DATASET_ROOT}
            RESULT_VARIABLE tiny_success
            ERROR_VARIABLE tiny_error
    )
    if(tiny_success)
        message(STATUS "Tiny ImageNet-200 successfully downloaded and converted.")
    else()
        message(FATAL_ERROR "Error downloading or converting Tiny ImageNet-200:\n${tiny_error}")
    endif()
endif()
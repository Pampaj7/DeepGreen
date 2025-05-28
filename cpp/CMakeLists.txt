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
    set(CMAKE_CUDA_ARCHITECTURES 89) # Adatta alla tua GPU:
    # 89 per L40S
    # 86 per RTX 30xx
    # 75 per RTX 20xx
    # 61 per GTX 10xx
    if (WIN32)
        # Aggiungi path a nvxt3 (profiler di cuda) che LibTorch non riesce ancora a trovare (v.2.7.0)
        # Da inserire prima di "find_package(Torch REQUIRED)"
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
        set(NVTX3_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include/nvtx3")
        include_directories(${NVTX3_INCLUDE_DIR})
    elseif (LINUX)
        set(CUDA_TOOLKIT_ROOT_DIR "/home/marcopaglio/tools/cuda-12.8")
        set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
    endif ()
    enable_language(CUDA)

    # Configure LibTorch
    if (WIN32)
        set(LIBTORCH_SPEC "libtorch_win32_cuda")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.0%2Bcu128.zip")
    elseif (LINUX)
        set(LIBTORCH_SPEC "libtorch_linux_cuda")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip")
    endif ()
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

# set output of FetchContent before using the command
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/tools/libtorch")

# Download LibTorch
include(FetchContent)
FetchContent_Declare(
        ${LIBTORCH_SPEC}
        URL ${LIBTORCH_URL}
        # the following option is necessary when cmake version is 3.23 or lower
        # it defines the default behavior of cmake >= 3.24, as recommended at
        # https://cmake.org/cmake/help/latest/policy/CMP0135.html
        DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_MakeAvailable(${LIBTORCH_SPEC})

# Aggiungi path a versione build di LibTorch
set(CMAKE_PREFIX_PATH "${FETCHCONTENT_BASE_DIR}/${LIBTORCH_SPEC}-src")

find_package(Torch REQUIRED)
MESSAGE(STATUS "LibTorch found.")


# Mettere dopo "find_package(Torch REQUIRED)"
if (CUDA_ENABLED AND WIN32)
    # Workaround per simulare CUDA::nvToolsExt con NVTX3 (header-only)
    if(NOT TARGET CUDA::nvToolsExt)
        message(STATUS "CUDA::nvToolsExt non trovato: creo un target fittizio con NVTX3 header-only")
        add_library(nvtx3_dummy INTERFACE)
        target_include_directories(nvtx3_dummy INTERFACE "${NVTX3_INCLUDE_DIR}")
        add_library(CUDA::nvToolsExt ALIAS nvtx3_dummy)
    endif()
endif ()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(resnet18_cifar100 src/train_cifar100.cpp)
set(TARGETS
        resnet18_cifar100
)


foreach (target ${TARGETS})
    target_link_libraries(${target} ${OpenCV_LIBS})

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
                CUDA_STANDARD 17
                CXX_STANDARD 17 # NECESSARIO altrimenti fallisce
        )
    endif ()
endforeach ()

add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
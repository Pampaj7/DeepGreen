cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(DeepGreen_cpp)

# set output of FetchContent before using the command
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/tools/libtorch")

if (WIN32)
    # Configure OpenCV
    # set(OpenCV_DIR "C:/Program Files (x86)/OpenCV/opencv_4.10.0/build")
    # or add "path/to/openCV/build/bin" to PATH env var

    # Configure LibTorch
    set(LIBTORCH_SPEC "libtorch_win32")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.7.0%2Bcpu.zip")
elseif (LINUX)
    # Configure OpenCV
    set(OpenCV_DIR "/home/marcopaglio/tools/opencv-4.11.0/build")

    # Configure LibTorch
    set(LIBTORCH_SPEC "libtorch_linux")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip")
endif ()

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

set(CMAKE_PREFIX_PATH "${FETCHCONTENT_BASE_DIR}/${LIBTORCH_SPEC}-src")

find_package(Torch REQUIRED)
MESSAGE(STATUS "LibTorch found.")

find_package(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV found.")
include_directories(${OpenCV_INCLUDE_DIRS})


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
endforeach ()

add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
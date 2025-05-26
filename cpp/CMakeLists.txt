cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(DeepGreen_cpp)

if (WIN32)
    # set(OpenCV_DIR "C:/Program Files (x86)/OpenCV/opencv_4.10.0/build")
    # or add "path/to/openCV/build/bin" to PATH env var
elseif (LINUX)
    set(OpenCV_DIR "/home/marcopaglio/tools/opencv-4.11.0/build")
endif ()

find_package(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV found.")
include_directories(${OpenCV_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 17)


add_executable(resnet18_cifar100 src/train_cifar100.cpp)
set(TARGETS
        resnet18_cifar100
)

foreach (target ${TARGETS})
    target_link_libraries(${target} ${OpenCV_LIBS})
endforeach ()

add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
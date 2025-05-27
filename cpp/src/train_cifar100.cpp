#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>


int main() {
    try {
        torch::DeviceType device_type;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Training on GPU." << std::endl;
            device_type = torch::kCUDA;
        } else {
            std::cout << "Training on CPU." << std::endl;
            device_type = torch::kCPU;
        }
        torch::Device device(device_type);

        torch::Tensor tensor = torch::rand({2, 3});
        std::cout << tensor << std::endl;


        std::string imageRelativePath = "/../data/test/01014.png";

        std::stringstream imgFullPathStream;
        imgFullPathStream << PROJECT_SOURCE_DIR << imageRelativePath;
        cv::Mat image = cv::imread(imgFullPathStream.str(), cv::IMREAD_COLOR);
        if (image.empty()) {
            CV_Error(-1, "Failed to load image at: " + imgFullPathStream.str());
        }
        imgFullPathStream.str(std::string());


        std::cout << "Dimensioni: " << image.cols << "x" << image.rows << std::endl;
    }
    catch (const cv::Exception& e) {
        const char* err_msg = e.what();
        std::cout << "exception caught: " << err_msg << std::endl;
        return e.code;
    }

    return 0;
}


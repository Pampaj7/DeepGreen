#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    try {
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
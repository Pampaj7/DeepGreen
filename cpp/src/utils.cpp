#include "utils.h"

#include <string>
#include <iostream>
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

std::string Utils::join_paths(std::string head, const std::string& tail)
{
    if (head.back() != '/' && tail.front() != '/')
    {
        head.push_back('/');
    }
    head += tail;
    return head;
}

void Utils::removeFileIfExists(const std::string& fullPathName)
{
    if (std::filesystem::exists(fullPathName) && !std::filesystem::is_directory(fullPathName))
        std::filesystem::remove(fullPathName);
}

std::string Utils::makeWindowsLongPathIfNeeded(const std::string& input_path) {
#ifdef _WIN32
    // backslash convertion
    std::string path = input_path;
    std::replace(path.begin(), path.end(), '/', '\\');

    try {
        // absolut path convertion
        const std::filesystem::path abs_path = std::filesystem::absolute(path);
        const std::string abs_path_str = abs_path.string();

        // prefix addition (if needed)
        if (abs_path_str.length() >= MAX_PATH && path.rfind(R"(\\?\)", 0) != 0)
                return R"(\\?\)" + abs_path_str;
        return abs_path_str;

    } catch (const std::exception& e) {
        std::cerr << "Error while converting to Windows usable long path: " << e.what() << '\n';
        return path;
    }
#else
    return input_path;
#endif
}

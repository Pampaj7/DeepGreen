#ifndef UTILS_H
#define UTILS_H
#include <string>


namespace Utils {

    std::string join_paths(std::string head, const std::string& tail);

    /**
     * On Windows systems the maximum path length is limited to 260 characters (MAX_PATH limit)
     * and some APIs (e.g. OpenCV) could fail, unless:
     * - the prefix \\?\ is added to the path, to deactivate the default parsing and using path until 32.767 characters
     * - backslash \ are used (instead of slash / ones)
     * - the path is absolute
     *
     * This function makes a long path usable on Windows system, if it is more than 260 characters,
     * and does nothing on all other systems.
     *
     * @param input_path the path to adjust to be used on Windows systems.
     * @return the path adjusted on Windows; otherwise, the input path.
     */
    std::string makeWindowsLongPathIfNeeded(const std::string& input_path);

};



#endif //UTILS_H

#ifndef UTILS_H
#define UTILS_H
#include <string>


namespace Utils {

    std::string join_paths(std::string head, const std::string& tail);

    std::string join_paths_as_absolute_path(const std::string& head, const std::string& tail);

};



#endif //UTILS_H

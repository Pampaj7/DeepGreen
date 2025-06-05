#include "utils.h"

#include <string>
#include <vector>
#include <sstream>
#include <iostream>

std::string Utils::join_paths(std::string head, const std::string& tail)
{
    if (head.back() != '/' && tail.front() != '/')
    {
        head.push_back('/');
    }
    head += tail;
    return head;
}

std::vector<std::string> split(const std::string& str, const char delim) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

std::string join(const std::vector<std::string>& parts, const char delim) {
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i) {
        result += parts[i];
        if (i + 1 < parts.size()) {
            result += delim;
        }
    }
    return result;
}

std::string Utils::join_paths_as_absolute_path(const std::string& head, const std::string& tail) {
    const bool isAbsolute = !head.empty() && head.front() == '/';

    std::vector<std::string> headParts = split(head, '/');
    std::vector<std::string> tailParts = split(tail, '/');

    auto it = tailParts.begin();
    while (it != tailParts.end() && *it == "..") {
        if (!headParts.empty()) {
            headParts.pop_back();
        }
        it = tailParts.erase(it);
    }

    std::string newHead = join(headParts, '/');
    std::string newTail = join(tailParts, '/');

    std::string result = isAbsolute ? "/" : "";
    result += newHead;
    if (!newHead.empty() && !newTail.empty())
        result += '/';
    result += newTail;

    return result;
}
#include "utils.h"

std::string Utils::join_paths(std::string head, const std::string& tail)
{
    if (head.back() != '/')
    {
        head.push_back('/');
    }
    head += tail;
    return head;
}

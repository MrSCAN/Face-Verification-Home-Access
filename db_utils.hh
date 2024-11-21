#ifndef DB_UTILS_H
#define DB_UTILS_H

#include <vector>
#include <string>

std::vector<std::pair<std::string, std::vector<float>>>  load_from_database();
void save_to_database(const std::vector<float> &descriptor, const std::string &name);

#endif // DB_UTILS_H

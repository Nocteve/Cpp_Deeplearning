std::string dir_name = entry.path().filename().string();
int label = std::isdigit(dir_name.back()) ? dir_name.back() - '0' : -1; // 或处理错误
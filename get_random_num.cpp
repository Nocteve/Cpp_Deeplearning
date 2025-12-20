#include <iostream>
#include <random>  // 核心随机数库
#include <chrono>  // 用于获取时间种子

int main() {
    // 1. 创建随机数引擎（生成器）
    // 使用当前时间作为种子，确保每次运行结果不同
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    
    // 2. 创建分布对象（定义随机数的范围和分布规律）
    
    // 生成均匀分布的整数 [0, 99]
    std::uniform_int_distribution<int> int_distribution(0, 99);
    std::cout << "0-99之间的随机整数: " << int_distribution(generator) << std::endl;
    
    // 生成均匀分布的浮点数 [0.0, 1.0)
    std::uniform_real_distribution<double> real_distribution(0.0, 1.0);
    std::cout << "0.0-1.0之间的随机浮点数: " << real_distribution(generator) << std::endl;
    
    // 生成正态分布的随机数（均值=0.0，标准差=1.0）
    std::normal_distribution<double> normal_distribution(0.0, 1.0);
    std::cout << "正态分布随机数: " << normal_distribution(generator) << std::endl;
    
    // 3. 生成多个随机数
    std::cout << "\n5个随机整数: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << int_distribution(generator) << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
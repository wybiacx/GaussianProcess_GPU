#include <cuda_runtime_api.h>
#include <iostream>


#include "cpu_GP.h"
#include "cu_GP.cuh"
#include <chrono> // 用于统计时间

int main() {

    using namespace std::chrono;

    // 统计 CPU GP 的运行时间
    std::cout << "CPU GP" << std::endl;
    auto cpu_start = high_resolution_clock::now();
    test_cpu_GP();
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    std::cout << "CPU GP Time: " << cpu_duration << " ms" << std::endl;

    // 统计 GPU GP 的运行时间
    std::cout << "GPU GP" << std::endl;
    auto gpu_start = high_resolution_clock::now();
    test_gpu_GP();
    auto gpu_end = high_resolution_clock::now();
    auto gpu_duration = duration_cast<milliseconds>(gpu_end - gpu_start).count();
    std::cout << "GPU GP Time: " << gpu_duration << " ms" << std::endl;

    return 0;
}

#include <cuda_runtime_api.h>
#include <iostream>


#include "cpu_GP.h"
#include "cu_GP.cuh"
#include <chrono> // 用于统计时间

int main() {

    using namespace std::chrono;

    std::vector<int> dataset = {100, 500, 1000, 5000, 10000};
    // std::vector<int> dataset = {100, 500};

    warmup_gpu_GP();
    int i = 0;
    for (const auto &data_len : dataset) {
        std::cout << "Test Case: #" << i++ << std::endl;
        std::cout << "\tTrain Dataset Size: " << data_len << std::endl;
        // 统计 CPU GP 的运行时间

        auto cpu_start = high_resolution_clock::now();
        test_cpu_GP(data_len);
        auto cpu_end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<milliseconds>(cpu_end - cpu_start).count();
        std::cout << "\tCPU GPR Time: " << cpu_duration << " ms" << std::endl;

        // 统计 GPU GP 的运行时间

        auto gpu_start = high_resolution_clock::now();
        test_gpu_GP(data_len);
        auto gpu_end = high_resolution_clock::now();
        auto gpu_duration = duration_cast<milliseconds>(gpu_end - gpu_start).count();
        std::cout << "\tGPU GPR Time: " << gpu_duration << " ms" << std::endl;

    }


    return 0;
}

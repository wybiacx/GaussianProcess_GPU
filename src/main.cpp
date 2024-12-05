#include <cuda_runtime_api.h>
#include <iostream>
// #include "test_kernel.cuh"

#include "cpu_GP.h"
#include "cu_GP.cuh"

int main() {


    test_cpu_GP();
    return 0;
}

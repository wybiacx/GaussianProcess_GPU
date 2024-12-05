//
// Created by Administrator on 24-12-5.
//

#include "test_kernel.cuh"

#include <cstdio>

__global__ void hello_test() {

    printf("Hello World\n");

}

void  test_kernel() {

    int threadsPerBlock = 8;
    int blocksPerGrid = 2;

    hello_test<<<blocksPerGrid, threadsPerBlock>>>();

    cudaDeviceSynchronize();
}

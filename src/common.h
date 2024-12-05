//
// Created by Administrator on 24-12-6.
//

#ifndef COMMON_H
#define COMMON_H


#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
    printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
    printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1);                                                            \
    }                   \
}

#endif //COMMON_H

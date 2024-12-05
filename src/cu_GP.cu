//
// Created by Administrator on 24-12-6.
//

#include "cu_GP.cuh"

#include <iostream>
#include <ostream>
#include <cuda_runtime.h>
#include "common.h"
#include <cusolverDn.h>


__device__ double rbf_kernel_cuda(const double *x1, const double *x2, const int dim, const double length_scale) {
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        double diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return exp(-0.5 * sum / (length_scale * length_scale));
}

__global__ void compute_covariance_matrix_cuda(const double *X1, const double *X2, double *K, const int n, const int m, const int dim, const double length_scale) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        K[i * n + j]= rbf_kernel_cuda(&X1[i * dim], &X2[j * dim], dim, length_scale);
    }
}

__global__ void compute_covariance_matrix_cuda_with_sigma(const double *X1, const double *X2, double *K, const int n,
    const int m, const int dim, const double length_scale, const double sigma) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        K[i * n + j]= rbf_kernel_cuda(&X1[i * dim], &X2[j * dim], dim, length_scale);
        if (i == j) K[i * n + j] += sigma;
    }
}

__global__ void mat_vec_multiply_cuda(const double *mat, const double *vec, const int n, double *alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < n; j++) {
            alpha[i] += mat[i * n + j] * vec[j];
        }
    }

}

__global__ void calculate_y_star_cuda(const double *d_K_star, const double *d_alpha, const int n) {
}

vector<double> gpu_gaussian_process_regression(const vector<vector<double>> &X_train, const vector<double> &y_train,
                                               const vector<vector<double>> &X_test, double length_scale, double sigma_n) {
    if (X_train.size() != y_train.size()) {
        cerr << "X_train.size() != y_train.size()" << endl;
        exit(-1);
    }
    if (X_train.size() <= 0 || X_test.size() <= 0 || y_train.size() <= 0) {
        cerr << "X_train.size() <= 0 || X_train.size() <= 0" << endl;
        exit(-1);
    }

    if (X_train[0].size() <= 0) {
        cerr << "X_train[0].size() <= 0" << endl;
        exit(-1);
    }

    const int element_len = X_train[0].size();
    const int total_elements = X_train.size() * element_len;

    double *flattened_data;

    CHECK(cudaHostAlloc(&flattened_data, sizeof(double) * total_elements, cudaHostAllocDefault));

    size_t index = 0;
    for (const auto &x : X_train) {
        memcpy(flattened_data + index * element_len, x.data(), element_len * sizeof(double));
        index++;
    }

    double *d_X_train, *d_K, *d_y_train;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X_train), sizeof(double) * total_elements));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) *  total_elements * total_elements));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y_train), sizeof(double) * total_elements));

    CHECK(cudaMemcpy(d_X_train, flattened_data, sizeof(double) * total_elements, cudaMemcpyHostToDevice));

    dim3 threadPerBlock(32, 32);
    dim3 blockPerGrid((total_elements + threadPerBlock.x - 1) / threadPerBlock.x, (total_elements + threadPerBlock.y - 1) / threadPerBlock.y);


    // 计算协方差矩阵 K
    // 协方差矩阵加噪声 sigma ^ 2 * I
    compute_covariance_matrix_cuda_with_sigma<<<blockPerGrid, threadPerBlock>>>(
        d_X_train, d_X_train, d_K,
        total_elements, total_elements,
        element_len, length_scale, sigma_n
    );


    // 计算协方差矩阵的逆 K^-1

    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    int *d_pivot;
    int *d_info;
    double *d_work;
    int work_size;

    CHECK(cudaMalloc((void**)&d_pivot, total_elements * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_info, sizeof(int)));

    cusolverDnDgetrf_bufferSize(cusolver_handle, total_elements, total_elements, d_K, total_elements, &work_size);
    cudaMalloc((void**)&d_work, work_size * sizeof(double));

    cusolverDnDgetrf(cusolver_handle, total_elements, total_elements, d_K, total_elements, d_work, d_pivot, d_info);

    cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N, total_elements, total_elements, d_K, total_elements, d_pivot, d_K, total_elements, d_info);


    // 计算协方差矩阵与y_train的相乘
    // K^-1 * y_train -> alpha
    double *d_alpha;
    CHECK(cudaMalloc((void**)&d_alpha, total_elements * sizeof(double)));

    dim3 threadPerBlock2(32);
    dim3 blockPerGrid2((total_elements + threadPerBlock2.x - 1) / threadPerBlock2.x);


    mat_vec_multiply_cuda<<<blockPerGrid2, threadPerBlock2>>>(d_K, d_y_train, total_elements, d_alpha);


    double *d_K_star;
    int predict_size = X_test.size();
    CHECK(cudaMalloc((void**)&d_K_star, total_elements * predict_size * sizeof(double)));

    dim3 threadPerBlock3(32, 32);
    dim3 blockPerGrid3((predict_size + threadPerBlock3.x - 1) / threadPerBlock3.x, (total_elements + threadPerBlock3.y - 1) / threadPerBlock3.y);


    // compute_covariance_matrix_cuda<<<blockPerGrid3, threadPerBlock3>>>()

    // 计算X_train与X_test之间的协方差矩阵K_*

    // 计算预测值 y_*

    vector<double> y_start(total_elements, 0);
    return y_start;

}

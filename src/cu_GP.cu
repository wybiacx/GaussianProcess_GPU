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
        K[i * m + j]= rbf_kernel_cuda(&X1[i * dim], &X2[j * dim], dim, length_scale);
    }
}

__global__ void compute_covariance_matrix_cuda_with_sigma(const double *X1, const double *X2, double *K, const int n,
    const int m, const int dim, const double length_scale, const double sigma) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        K[i * m + j]= rbf_kernel_cuda(&X1[i * dim], &X2[j * dim], dim, length_scale);
        if (i == j) K[i * m + j] += sigma;
    }
}

__global__ void mat_vec_multiply_cuda(const double *mat, const double *vec, const int n, const int m, double *alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < m; j++) {
            alpha[i] += mat[i * m + j] * vec[j];
        }
    }

}

__global__ void initialize_identity_matrix(double *I, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        I[row * n + col] = (row == col) ? 1.0 : 0.0;
    }
}

// 函数：将 GPU 数组拷贝到 CPU 并打印
void printDeviceArrayToHost(const double* d_array, size_t size) {
    // 在 CPU 上分配内存
    double* h_array = new double[size];

    // 将数据从设备端（GPU）拷贝到主机端（CPU）
    CHECK(cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost));

    // 打印数组的值
    std::cout << "Array values on host (copied from device):" << std::endl;
    for (size_t i = 0; i < size; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // 释放 CPU 上分配的内存
    delete[] h_array;
}

void printDeviceArrayToHostMatrix(const double* d_array, size_t size, int n, int m) {
    // 在 CPU 上分配内存
    double* h_array = new double[size];

    // 将数据从设备端（GPU）拷贝到主机端（CPU）
    CHECK(cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost));

    // 打印数组的值
    std::cout << "Array values on host (copied from device):" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            std::cout << h_array[i * m + j] << " ";
        }
        std::cout << std::endl;
    }


    // 释放 CPU 上分配的内存
    delete[] h_array;
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

    int device = 0;
    CHECK(cudaSetDevice(device));

    const int element_len = X_train[0].size();
    const int X_train_count = X_train.size();

    const int X_test_count = X_test.size();

    double *flattened_train, *flattened_test;

    CHECK(cudaHostAlloc(&flattened_train, sizeof(double) * X_train_count * element_len, cudaHostAllocDefault));
    CHECK(cudaHostAlloc(&flattened_test, sizeof(double) * X_test_count * element_len, cudaHostAllocDefault));

    size_t index = 0;
    for (const auto &x : X_train) {
        memcpy(flattened_train + index * element_len, x.data(), element_len * sizeof(double));
        index++;
    }

    index = 0;
    for (const auto &x : X_test) {
        memcpy(flattened_test + index * element_len, x.data(), element_len * sizeof(double));
        index++;
    }

    vector<double> y_start(X_test_count, 0);
    double *d_X_train, *d_K, *d_y_train, *d_X_test, *d_K_backup, *d_K_inv, *d_alpha, *d_K_star, *d_y_star;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X_train), sizeof(double) * X_train_count * element_len));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X_test), sizeof(double) * X_test_count * element_len));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) *  X_train_count * X_train_count * element_len));
    CHECK(cudaMalloc((void**)&d_K_backup, sizeof(double) * X_train_count * X_train_count));
    CHECK(cudaMalloc((void**)&d_K_inv, sizeof(double) * X_train_count * X_train_count));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y_train), sizeof(double) * X_train_count));
    CHECK(cudaMalloc((void**)&d_K_star, X_train_count * X_test_count * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_alpha, X_train_count * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_y_star, X_test_count * sizeof(double)));

    CHECK(cudaMemcpy(d_y_train, y_train.data(), sizeof(double) * X_train_count, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_X_train, flattened_train, sizeof(double) * X_train_count * element_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_X_test, flattened_test, sizeof(double) * X_test_count * element_len, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((X_train_count + threadsPerBlock.x - 1) / threadsPerBlock.x, (X_train_count + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 计算协方差矩阵 K
    // 协方差矩阵加噪声 sigma ^ 2 * I
    compute_covariance_matrix_cuda_with_sigma<<<blocksPerGrid, threadsPerBlock>>>(
        d_X_train, d_X_train, d_K,
        X_train_count, X_train_count,
        element_len, length_scale, sigma_n
    );

    cudaDeviceSynchronize();

    // --------------------------------------------------------------------------------------------
    // 计算协方差矩阵的逆 K^-1

    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    int *d_pivot;
    int *d_info;
    double *d_work;
    int work_size;
    int h_info;
    CHECK(cudaMalloc((void**)&d_pivot, X_train_count * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_info, sizeof(int)));

    CHECK(cudaMemcpy(d_K_backup, d_K, sizeof(double) * X_train_count * X_train_count, cudaMemcpyDeviceToDevice));

    // 初始化单位矩阵
    initialize_identity_matrix<<<blocksPerGrid, threadsPerBlock>>>(d_K_inv, X_train_count);
    cudaDeviceSynchronize();

    cusolverDnDgetrf_bufferSize(cusolver_handle, X_train_count, X_train_count, d_K, X_train_count, &work_size);
    CHECK(cudaMalloc((void**)&d_work, work_size * sizeof(double)));

    cusolverDnDgetrf(cusolver_handle, X_train_count, X_train_count, d_K, X_train_count, d_work, d_pivot, d_info);

    cudaDeviceSynchronize();

    CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "LU decomposition failed. Info: " << h_info << std::endl;
        exit(-1);
    }

    cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N, X_train_count, X_train_count, d_K, X_train_count, d_pivot, d_K_inv, X_train_count, d_info);

    cudaDeviceSynchronize();

    // 检查求解是否成功
    CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "Matrix inversion failed. Info: " << h_info << std::endl;
        exit(-1);
    }
    //---------------------------------------------------------------------------------------

    // 计算协方差矩阵与y_train的相乘
    // K^-1 * y_train -> alpha

    dim3 threadPerBlock2(32);
    dim3 blockPerGrid2((X_train_count + threadPerBlock2.x - 1) / threadPerBlock2.x);

    // [n, n] x [n, 1] -> [n, 1]
    mat_vec_multiply_cuda<<<blockPerGrid2, threadPerBlock2>>>(d_K_inv, d_y_train, X_train_count, X_train_count, d_alpha);
    cudaDeviceSynchronize();

    // 计算X_test与X_train之间的协方差矩阵K_*

    dim3 threadPerBlock3(32, 32);
    dim3 blockPerGrid3((X_test_count + threadPerBlock3.x - 1) / threadPerBlock3.x, (X_train_count + threadPerBlock3.y - 1) / threadPerBlock3.y);
    // [m, n]
    compute_covariance_matrix_cuda<<<blockPerGrid3, threadPerBlock3>>>(d_X_test, d_X_train, d_K_star,
        X_test_count, X_train_count, element_len, length_scale);

    cudaDeviceSynchronize();


    // 计算预测值 y_*
    dim3 threadPerBlock4(32);
    dim3 blockPerGrid4((X_test_count + threadPerBlock4.x - 1) / threadPerBlock4.x);

    // [m,n] x [n, 1] -> [m, 1]
    mat_vec_multiply_cuda<<<blockPerGrid4, threadPerBlock4>>>(d_K_star, d_alpha, X_test_count,X_train_count, d_y_star);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(y_start.data(), d_y_star, X_test_count * sizeof(double), cudaMemcpyDeviceToHost));


    CHECK(cudaFree(d_K_star));
    CHECK(cudaFree(d_X_test));
    CHECK(cudaFree(d_X_train));
    CHECK(cudaFree(d_alpha));
    CHECK(cudaFree(d_info));
    CHECK(cudaFree(d_pivot));
    CHECK(cudaFree(d_work));
    CHECK(cudaFree(d_y_star));
    CHECK(cudaFree(d_y_train));
    CHECK(cudaFree(d_K));
    CHECK(cudaFree(d_K_backup));
    CHECK(cudaFree(d_K_inv));
    CHECK(cudaFreeHost(flattened_train));
    CHECK(cudaFreeHost(flattened_test));
    cusolverDnDestroy(cusolver_handle);

    return y_start;

}

void warmup_gpu_GP() {
    int data_len = 100;
    // 示例训练数据（多维输入）
    vector<vector<double>> X_train ;  // 训练点
    vector<double> y_train;  // 对应的输出

    srand(time(NULL));

    int n = data_len;
    int input_dim = 10;

    for (int i = 0; i < n; i++) {
        vector<double> input;
        double output = 0;
        for (int j = 0; j < input_dim; j++) {
            double elem = i + j;
            input.push_back(elem);
            output += input[j];
        }
        X_train.push_back(input);
        y_train.push_back(output);
    }

    // 示例测试数据（多维输入）
    vector<vector<double>> X_test;  // 测试点
    vector<double> y_test;

    int test_len = 10;
    for (int i = 0; i < test_len; i++) {
        vector<double> input;
        double output = 0;
        for (int j = 0; j < input_dim; j++) {
            double elem = i + j + 0.5;
            input.push_back(elem);
            output += input[j];
        }
        X_test.push_back(input);
        y_test.push_back(output);
    }



    double length_scale = 1.0;  // RBF核的长度尺度
    double sigma_n = 1e-2;  // 噪声的标准差

    // 进行高斯过程回归预测
    try {
        vector<double> y_star = gpu_gaussian_process_regression(X_train, y_train, X_test, length_scale, sigma_n);

        // 输出预测结果
        // cout << "Predicted values: " << endl;
        // for (double val : y_star) {
        //     cout << val << endl;
        // }

        // double error = 0;
        // for (int i = 0; i < X_test.size(); ++i) {
        //     error += fabs(y_test[i] - y_star[i]);
        // }
        // cout << "Total Error: " << error << endl;
        cout << "GPU WarmUp Successes. Using Device 0." << endl;

    } catch (const exception& e) {
        cerr << "Warm up GPU Error: " << e.what() << endl;
    }
}


void test_gpu_GP(int data_len) {
    // 示例训练数据（多维输入）
    vector<vector<double>> X_train ;  // 训练点
    vector<double> y_train;  // 对应的输出

    srand(time(NULL));

    int n = data_len;
    int input_dim = 10;

    for (int i = 0; i < n; i++) {
        vector<double> input;
        double output = 0;
        for (int j = 0; j < input_dim; j++) {
            double elem = i + j;
            input.push_back(elem);
            output += input[j];
        }
        X_train.push_back(input);
        y_train.push_back(output);
    }

    // 示例测试数据（多维输入）
    vector<vector<double>> X_test;  // 测试点
    vector<double> y_test;

    int test_len = 10;
    for (int i = 0; i < test_len; i++) {
        vector<double> input;
        double output = 0;
        for (int j = 0; j < input_dim; j++) {
            double elem = i + j + 0.5;
            input.push_back(elem);
            output += input[j];
        }
        X_test.push_back(input);
        y_test.push_back(output);
    }



    double length_scale = 1.0;  // RBF核的长度尺度
    double sigma_n = 1e-2;  // 噪声的标准差

    // 进行高斯过程回归预测
    try {
        vector<double> y_star = gpu_gaussian_process_regression(X_train, y_train, X_test, length_scale, sigma_n);

        // 输出预测结果
        // cout << "Predicted values: " << endl;
        // for (double val : y_star) {
        //     cout << val << endl;
        // }

        // double error = 0;
        // for (int i = 0; i < X_test.size(); ++i) {
        //     error += fabs(y_test[i] - y_star[i]);
        // }
        // cout << "Total Error: " << error << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

}

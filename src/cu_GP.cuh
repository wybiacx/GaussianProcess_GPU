//
// Created by Administrator on 24-12-6.
//

#ifndef CU_GP_CUH
#define CU_GP_CUH

#include <vector>
using namespace std;

//CUDA rbf kernel

__device__ double rbf_kernel_cuda(const double *x1, const double *x2, const int dim, const double length_scale);

__global__ void compute_covariance_matrix_cuda(const double *X1, const double *X2, double *K, const int n, const int m,
    const int dim, const double length_scale);

__global__ void compute_covariance_matrix_cuda_with_sigma(const double *X1, const double *X2, double *K, const int n, const int m,
    const int dim, const double length_scale, const double sigma);

__global__ void mat_vec_multiply_cuda(const double *mat, const double *vec, const int n, const int m, double *alpha);

__global__ void initialize_identity_matrix(double* I, int n);


vector<double> gpu_gaussian_process_regression(const vector<vector<double>>& X_train, const vector<double>& y_train,
                                           const vector<vector<double>>& X_test, double length_scale, double sigma_n);

void test_gpu_GP(int data_len);

void warmup_gpu_GP();

#endif //CU_GP_CUH

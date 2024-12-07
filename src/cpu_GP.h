//
// Created by Administrator on 24-12-5.
//

#ifndef CPU_GP_H
#define CPU_GP_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
using namespace std;

// 定义核函数（RBF核）
double rbf_kernel(const vector<double>& x1, const vector<double>& x2, double length_scale);

// 计算协方差矩阵 K
vector<vector<double>> compute_covariance_matrix(const vector<vector<double>>& X, double length_scale);

// 矩阵求逆（高斯-约旦消元法）
bool inverse_matrix(vector<vector<double>>& mat);

// 矩阵与向量的乘法
vector<double> mat_vec_multiply(const vector<vector<double>>& mat, const vector<double>& vec);

vector<double> cpu_gaussian_process_regression(const vector<vector<double>>& X_train, const vector<double>& y_train,
                                           const vector<vector<double>>& X_test, double length_scale, double sigma_n);

// 测试

void test_cpu_GP(int data_len);

#endif //CPU_GP_H

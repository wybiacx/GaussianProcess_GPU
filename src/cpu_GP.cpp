//
// Created by Administrator on 24-12-5.
//

#include "cpu_GP.h"

// 定义核函数（RBF核）
double rbf_kernel(const vector<double>& x1, const vector<double>& x2, double length_scale) {
    double sum = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        double diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return exp(-0.5 * sum / (length_scale * length_scale));
}

// 计算协方差矩阵 K
vector<vector<double>> compute_covariance_matrix(const vector<vector<double>>& X, double length_scale) {
    size_t n = X.size();
    vector<vector<double>> K(n, vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            K[i][j] = rbf_kernel(X[i], X[j], length_scale);
        }
    }

    return K;
}

// 矩阵求逆（高斯-约旦消元法）
bool inverse_matrix(vector<vector<double>>& mat) {
    size_t n = mat.size();
    vector<vector<double>> augmented(n, vector<double>(2 * n));

    // 构造增广矩阵 [A | I]
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            augmented[i][j] = mat[i][j];
            augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    // 高斯-约旦消元法
    for (size_t i = 0; i < n; ++i) {
        double diag = augmented[i][i];
        if (fabs(diag) < 1e-10) {
            return false; // 矩阵不可逆
        }
        for (size_t j = 0; j < 2 * n; ++j) {
            augmented[i][j] /= diag;
        }

        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                double factor = augmented[j][i];
                for (size_t k = 0; k < 2 * n; ++k) {
                    augmented[j][k] -= factor * augmented[i][k];
                }
            }
        }
    }

    // 提取右半部分作为逆矩阵
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            mat[i][j] = augmented[i][j + n];
        }
    }

    return true;
}

// 矩阵与向量的乘法
vector<double> mat_vec_multiply(const vector<vector<double>>& mat, const vector<double>& vec) {
    size_t n = mat.size();
    vector<double> result(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }

    return result;
}

// 高斯过程回归预测
vector<double> cpu_gaussian_process_regression(const vector<vector<double>>& X_train, const vector<double>& y_train,
                                           const vector<vector<double>>& X_test, double length_scale, double sigma_n) {
    size_t n_train = X_train.size();
    size_t n_test = X_test.size();

    // 计算训练数据的协方差矩阵
    vector<vector<double>> K = compute_covariance_matrix(X_train, length_scale);

    // 添加噪声项 sigma^2 * I
    for (size_t i = 0; i < n_train; ++i) {
        K[i][i] += sigma_n;
    }

    // 求逆 K^-1
    if (!inverse_matrix(K)) {
        throw runtime_error("Matrix is singular and cannot be inverted.");
    }

    // 计算 K^-1 * y_train
    vector<double> alpha = mat_vec_multiply(K, y_train);

    // 计算测试数据与训练数据之间的协方差矩阵 K_*
    vector<vector<double>> K_star(n_test, vector<double>(n_train, 0.0));
    for (size_t i = 0; i < n_test; ++i) {
        for (size_t j = 0; j < n_train; ++j) {
            K_star[i][j] = rbf_kernel(X_test[i], X_train[j], length_scale);
        }
    }

    // 计算预测值 y_*
    vector<double> y_star(n_test, 0.0);
    for (size_t i = 0; i < n_test; ++i) {
        for (size_t j = 0; j < n_train; ++j) {
            y_star[i] += K_star[i][j] * alpha[j];
        }
    }

    return y_star;
}


void test_cpu_GP() {
    // 示例训练数据（多维输入）
    vector<vector<double>> X_train ;  // 训练点
    vector<double> y_train;  // 对应的输出

    srand(time(NULL));

    int n = 100;
    int input_dim = 1;

    for (int i = 0; i < n; i++) {
        vector<double> input;
        double output = 0;
        for (int j = 0; j < input_dim; j++) {
            double elem = i;
            input.push_back(elem);
            output += input[j] * input[j];
        }
        X_train.push_back(input);
        y_train.push_back(output);
    }

    // 示例测试数据（多维输入）
    vector<vector<double>> X_test = {{1.5}, {2.5}, {3.5}, {98.5}};  // 测试点

    vector<double> y_test;
    for (int i = 0; i < X_test.size(); i++) {
        double output = 0;
        for (int j = 0; j < input_dim; j++) {
            double elem = X_test[i][j];
            output += elem * elem;
        }
        y_test.push_back(output);
    }

    double length_scale = 1.0;  // RBF核的长度尺度
    double sigma_n = 1e-1;  // 噪声的标准差

    // 进行高斯过程回归预测
    try {
        vector<double> y_star = cpu_gaussian_process_regression(X_train, y_train, X_test, length_scale, sigma_n);

        // 输出预测结果
        cout << "Predicted values: " << endl;
        for (double val : y_star) {
            cout << val << endl;
        }

        double error = 0;
        for (int i = 0; i < X_test.size(); ++i) {
            error += fabs(y_test[i] - y_star[i]);
        }
        cout << "Total Error: " << error << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

}

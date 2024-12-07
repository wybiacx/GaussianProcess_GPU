## Gaussian Process Regression with GPU Acceleration

This project aims to accelerate Gaussian Process Regression (GPR) computations using NVIDIA GPUs. It includes both CPU-based and GPU-accelerated implementations of Gaussian Processes (GP), with the GPU version leveraging NVIDIA's cuSolver library to significantly enhance performance.

Both the CPU and GPU versions of GPR in this project are experimental, developed as part of an exploration to deepen the author’s understanding of CUDA. These implementations are not recommended for production environments without thorough testing and consideration.

### Performance Comparison

The following table compares the time costs for GPR computations on an Intel i7-10700 CPU paired with an RTX 2080 Ti GPU, and 32GB of RAM.

| Data Size | GPU (ms) | CPU (ms) |
|:---------:|:--------:|:--------:|
|    100    |    4     |    1     |
|    500    |    16    |   101    |
|   1000    |    29    |   1212   |
|   5000    |   1043   |  239928  |
|   10000   |   6073   | 1277158  |

### Graphical Representation

The graph below visually compares the time costs of GPR on GPU versus CPU. The vertical axis uses a logarithmic scale for better visualization of performance differences. 

When the data size is 100, the CPU takes 1 ms, which is barely visible in the graph.

![GPR Cost Compare](images/GPR_CPU_vs_GPU.png)

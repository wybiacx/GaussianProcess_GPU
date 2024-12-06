#enable_language(C CXX CUDA)
cmake_minimum_required(VERSION 3.30)
project(GP_CUDA CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

# 设置 CUDA Toolkit 的路径（根据实际安装路径调整）
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")




add_executable(GP_CUDA src/main.cpp
        src/cpu_GP.cpp
        src/cpu_GP.h
        src/cu_GP.cu
        src/cu_GP.cuh
        src/common.h)

set_target_properties(GP_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(GP_CUDA PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-I${CUDA_TOOLKIT_ROOT_DIR}/include>)

target_link_libraries(GP_CUDA
        cuda
        cusolver
        cublas
        cudart
)
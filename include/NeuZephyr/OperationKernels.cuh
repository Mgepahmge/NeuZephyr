//
// Created by Administrator on 24-11-11.
//

#ifndef OPERATIONKERNELS_CUH
#define OPERATIONKERNELS_CUH

#include "stdio.h"
#define TILE_SIZE 32

namespace NeuZephyr::Operator {
    __global__ void add_kernel(const float *a, const float *b, float *c, unsigned long long n);

    __global__ void sub_kernel(const float *a, const float *b, float *c, unsigned long long n);

    __global__ void GEMM_kernel(const float* A, const float* B, float* C, const unsigned long long M, const unsigned long long N, const unsigned long long K);

    __global__ void Transpose_kernel(const float* d_A, float* d_B, const unsigned int rows, const unsigned int cols);

    __global__ void ScalarMul_kernel (float* out, const float* in, const float num, unsigned long long n);
}

#endif //OPERATIONKERNELS_CUH

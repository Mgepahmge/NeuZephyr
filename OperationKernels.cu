//
// Created by Administrator on 24-11-11.
//

#include "OperationKernels.cuh"

__global__ void add_kernel(const float *a, const float *b, float *c, unsigned long long n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_kernel(const float *a, const float *b, float *c, unsigned long long n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void GEMM_kernel(const float* A, const float* B, float* C, const unsigned long long M, const unsigned long long N, const unsigned long long K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const unsigned long long row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for(int t = 0; t < K; t += TILE_SIZE) {
        if(row < M && t+threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row*K + t+threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if(col < N && t+threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t+threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int i = 0; i < TILE_SIZE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if(row < M && col < N)
        C[row*N + col] = sum;
}
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

__global__ void Transpose_kernel(const float* d_A, float* d_B, const unsigned int rows, const unsigned int cols)
{

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < rows && col < cols)
        tile[threadIdx.y][threadIdx.x] = d_A[row * cols + col];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f; // 填充越界部分为0

    __syncthreads();

    row = blockIdx.x * TILE_SIZE + threadIdx.y;
    col = blockIdx.y * TILE_SIZE + threadIdx.x;
    if (row < cols && col < rows)
        d_B[row * rows + col] = tile[threadIdx.x][threadIdx.y];
}
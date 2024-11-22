//
// Created by Administrator on 24-11-11.
//

#ifndef OPERATIONKERNELS_CUH
#define OPERATIONKERNELS_CUH

#include "stdio.h"
#define TILE_SIZE 32

namespace NeuZephyr::Operator {
    __global__ void add_kernel(const float* a, const float* b, float* c, unsigned long long n);

    __global__ void sub_kernel(const float* a, const float* b, float* c, unsigned long long n);

    __global__ void GEMM_kernel(const float* A, const float* B, float* C, const unsigned long long M,
                                const unsigned long long N, const unsigned long long K);

    __global__ void Transpose_kernel(const float* d_A, float* d_B, const unsigned int rows, const unsigned int cols);

    __global__ void ScalarMul_kernel(float* out, const float* in, const float num, unsigned long long n);

    __global__ void ScalarDiv_kernel(float* out, const float* in, const float num, unsigned long long n);

    __global__ void ScalarAdd_kernel(float* out, const float* in, const float num, unsigned long long n);

    __global__ void Negation_kernel(float* out, const float* in, unsigned long long n);

    __global__ void Recip_kernel(float* out, const float* in, unsigned long long n);

    __global__ void ReLU_kernel(float* out, const float* in, unsigned long long n);

    __global__ void ReLUBackward_kernel(float* A_grad, const float* A, const float* B_grad, unsigned long long n);

    __global__ void Sigmoid_kernel(float* out, const float* in, unsigned long long n);

    __global__ void SigmoidBackward_kernel(float* A_grad, const float* B, const float* B_grad, unsigned long long n);

    __global__ void Tanh_kernel(float* out, const float* in, unsigned long long n);

    __global__ void TanhBackward_kernel(float* A_grad, const float* B, const float* B_grad, unsigned long long n);

    __global__ void LeakyReLU_kernel(float* out, const float* in, unsigned long long n, float alpha = 0.01f);

    __global__ void LeakyReLUBackward_kernel(float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                             float alpha = 0.01f);

    __global__ void Swish_kernel(float* out, const float* in, unsigned long long n);

    __global__ void SwishBackward_kernel(float* A_grad, const float* A, const float* B, const float* B_grad,
                                         unsigned long long n);

    __global__ void ELU_kernel(float* out, const float* in, unsigned long long n, float alpha = 1.0f);

    __global__ void ELUBackward_kernel(float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                       float alpha = 1.0f);

    __global__ void HardSigmoid_kernel(float* out, const float* in, unsigned long long n, float alpha = 0.2f,
                                       float beta = 0.5f);

    __global__ void HardSigmoidBackward_kernel(float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                               float alpha = 0.2f, float beta = 0.5f);

    __global__ void HardSwish_kernel(float* out, const float* in, unsigned long long n, float alpha = 0.2f,
                                     float beta = 0.5f);

    __global__ void HardSwishBackward_kernel(float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                             float alpha = 0.2f, float beta = 0.5f);

    __global__ void ExpSum_kernel(float* out, const float* g_data, unsigned long long n);

    __global__ void Softmax_kernel(float* out, const float* in, float exp_sum_of_input, unsigned long long n);

    __global__ void SoftmaxJacobian_kernel(float* out, const float* in, unsigned long long n);

    __global__ void MSE_kernel(float* out, const float* predict, const float* real, unsigned long long n);

    __global__ void MSEBackward_kernel(float* out, const float* predict, const float* real, unsigned long long n);

    __global__ void SGD_kernel(float* data, const float* grad, const float lr, unsigned long long n);

    __global__ void BCE_kernel(float* out, const float* predict, const float* real, unsigned long long n);

    __global__ void BCEBackward_kernel(float* out, const float* predict, const float* real, unsigned long long n);

    __global__ void Momentum_kernel(float* output, const float* grad, const float* velocity, float beta, unsigned long long n);

    __global__ void AdaGrad_kernel(float* data, float* G, const float* grad, float lr, float eps, unsigned long long n);

    __global__ void RMSprop_kernel(float* data, float* v, const float* grad, const float lr, const float beta, const float eps, unsigned long long n);
}

#endif //OPERATIONKERNELS_CUH

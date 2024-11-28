#ifndef OPERATIONKERNELS_CUH
#define OPERATIONKERNELS_CUH

#include "stdio.h"
#include "utils.cuh"
#define TILE_SIZE 32

namespace NeuZephyr::Kernels {
    __global__ void MatrixAdd(const float* a, const float* b, float* c, unsigned long long n);

    __global__ void MatrixSub(const float* a, const float* b, float* c, unsigned long long n);

    __global__ void GeneralMatrixMul(const float* A, const float* B, float* C, const unsigned long long M,
                                     const unsigned long long N, const unsigned long long K);

    __global__ void Transpose(const float* d_A, float* d_B, const unsigned int rows, const unsigned int cols);

    __global__ void ScalarMul(float* out, const float* in, const float num, unsigned long long n);

    __global__ void ScalarDiv(float* out, const float* in, const float num, unsigned long long n);

    __global__ void ScalarAdd(float* out, const float* in, const float num, unsigned long long n);

    __global__ void Negation(float* out, const float* in, unsigned long long n);

    __global__ void Recip(float* out, const float* in, unsigned long long n);

    __global__ void RectifiedLinearUnit(float* out, const float* in, unsigned long long n);

    __global__ void ReLUBackward(float* A_grad, const float* A, const float* B_grad, unsigned long long n);

    __global__ void Sigmoid(float* out, const float* in, unsigned long long n);

    __global__ void SigmoidBackward(float* A_grad, const float* B, const float* B_grad, unsigned long long n);

    __global__ void Tanh(float* out, const float* in, unsigned long long n);

    __global__ void TanhBackward(float* A_grad, const float* B, const float* B_grad, unsigned long long n);

    __global__ void LeakyReLU(float* out, const float* in, unsigned long long n, float alpha = 0.01f);

    __global__ void LeakyReLUBackward(float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                      float alpha = 0.01f);

    __global__ void Swish(float* out, const float* in, unsigned long long n);

    __global__ void SwishBackward(float* A_grad, const float* A, const float* B, const float* B_grad,
                                  unsigned long long n);

    __global__ void ExponentialLinearUnit(float* out, const float* in, unsigned long long n, float alpha = 1.0f);

    __global__ void ELUBackward(float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                float alpha = 1.0f);

    __global__ void HardSigmoid(float* out, const float* in, unsigned long long n, float alpha = 0.2f,
                                float beta = 0.5f);

    __global__ void HardSigmoidBackward(float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                        float alpha = 0.2f, float beta = 0.5f);

    __global__ void HardSwish(float* out, const float* in, unsigned long long n, float alpha = 0.2f,
                              float beta = 0.5f);

    __global__ void HardSwishBackward(float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                      float alpha = 0.2f, float beta = 0.5f);

    __global__ void SummationExp(float* out, const float* g_data, unsigned long long n);

    __global__ void Softmax(float* out, const float* in, float exp_sum_of_input, unsigned long long n);

    __global__ void SoftmaxJacobian(float* out, const float* in, unsigned long long n);

    __global__ void MeanSquaredError(float* out, const float* predict, const float* real, unsigned long long n);

    __global__ void MSEBackward(float* out, const float* predict, const float* real, unsigned long long n);

    __global__ void StochasticGradientDescent(float* data, const float* grad, const float lr, unsigned long long n);

    __global__ void BinaryCrossEntropy(float* out, const float* predict, const float* real, unsigned long long n);

    __global__ void BCEBackward(float* out, const float* predict, const float* real, unsigned long long n);

    __global__ void Momentum(float* output, const float* grad, const float* velocity, float beta,
                             unsigned long long n);

    __global__ void AdaGrad(float* data, float* G, const float* grad, float lr, float eps, unsigned long long n);

    __global__ void RMSprop(float* data, float* v, const float* grad, const float lr, const float beta,
                            const float eps, unsigned long long n);

    __global__ void Adam(float* data, float* m, float* v, const float* grad, const float lr, const float beta1,
                         const float beta2, const float eps, const int t, unsigned long long n);

    __global__ void NAdam(float* data, float* m, float* m_modified, float* v, const float* grad, const float lr,
                          const float beta1, const float beta2, const float eps, const int t,
                          unsigned long long n);

    __global__ void AdaDelta(float* data, float* acc_delta, float* acc_grad, const float* grad,
                             const float rho, const float eps,
                             unsigned long long n);


}

#endif //OPERATIONKERNELS_CUH

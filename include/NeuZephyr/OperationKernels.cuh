/**
 * @file OperationKernels.cuh
 * @brief CUDA Kernel Definitions for High-Performance Tensor Operations
 *
 * This header file provides a comprehensive collection of CUDA kernel functions for
 * accelerated tensor computations, designed to support various mathematical operations,
 * neural network layers, activation functions, and optimization algorithms.
 *
 * @details
 * The kernel functions in this file are organized within the `nz::krnl` namespace
 * and cover a wide range of computational tasks:
 *
 * - **Matrix Operations**: Basic matrix arithmetic like addition, subtraction, multiplication,
 *   and transposition.
 * - **Element-wise Operations**: Scalar operations, negation, reciprocal calculations.
 * - **Activation Functions**:
 *   - Linear: ReLU, Leaky ReLU
 *   - Sigmoid Variants: Standard Sigmoid, Hard Sigmoid
 *   - Non-linear: Tanh, Swish, ELU
 * - **Backward Propagation Kernels**: Gradient computations for each activation function.
 * - **Loss Functions**: Mean Squared Error, Binary Cross-Entropy
 * - **Optimization Algorithms**: Stochastic Gradient Descent, Momentum, AdaGrad, RMSprop,
 *   Adam, NAdam, AdaDelta
 *
 * Kernels are designed for parallel execution on CUDA-enabled GPUs, leveraging high-performance
 * computing capabilities for efficient deep learning computations.
 *
 * @note
 * - All kernels utilize `unsigned long long` for size parameters to support large tensor dimensions.
 * - Most kernels operate on raw float pointers for maximum flexibility and performance.
 * - Kernel launch configurations (grid and block sizes) should be carefully managed to
 *   ensure optimal GPU utilization.
 *
 * @warning
 * These low-level CUDA kernel functions are intended for internal library implementation
 * and framework extension. End-users building neural network models SHOULD NOT directly
 * call these kernels. They are meant to be used exclusively by library developers
 * contributing to the internal functionality of the nz framework.
 *
 * @author
 * Mgepahmge (https://github.com/Mgepahmge)
 *
 * @date
 * 2024/12/07
 */
#ifndef OPERATIONKERNELS_CUH
#define OPERATIONKERNELS_CUH
#ifdef __CUDACC__

#include "stdio.h"
#include "utils.cuh"
#define TILE_SIZE 32

/**
 * @namespace nz::krnl
 * @brief High-Performance CUDA Kernel Implementations for Tensor Computations
 *
 * @details The nz::krnl namespace provides an extensive collection of CUDA kernel
 * functions optimized for accelerated tensor operations and deep learning computations.
 *
 * @section kernel_categories Kernel Function Categories
 *
 * The namespace encompasses several critical categories of computational kernels:
 *
 * @subsection matrix_ops Matrix Operations
 * - Matrix addition, subtraction
 * - General matrix multiplication
 * - Matrix transposition
 *
 * @subsection scalar_ops Scalar Operations
 * - Element-wise scalar multiplication
 * - Element-wise scalar division
 * - Element-wise scalar addition
 * - Negation
 * - Reciprocal calculations
 *
 * @subsection activation_funcs Activation Functions
 * Linear Activations:
 * - ReLU (Rectified Linear Unit)
 * - Leaky ReLU
 *
 * Non-linear Activations:
 * - Sigmoid
 * - Hard Sigmoid
 * - Tanh
 * - Swish
 * - Exponential Linear Unit (ELU)
 * - Hard Swish
 *
 * @subsection backward_props Backward Propagation Kernels
 * Gradient computation kernels for each activation function, supporting
 * efficient backpropagation in neural network training.
 *
 * @subsection loss_funcs Loss Functions
 * - Mean Squared Error (MSE)
 * - Binary Cross-Entropy (BCE)
 *
 * @subsection optimization_algos Optimization Algorithms
 * - Stochastic Gradient Descent (SGD)
 * - Momentum
 * - AdaGrad
 * - RMSprop
 * - Adam
 * - NAdam
 * - AdaDelta
 *
 * @note Performance Characteristics
 * - Designed for parallel execution on CUDA-enabled GPUs
 * - Utilizes `unsigned long long` for supporting large tensor dimensions
 * - Operates on raw float pointers for maximum performance and flexibility
 *
 * @warning
 * These low-level CUDA kernels are intended for internal library implementation.
 * End-users should NOT directly invoke these kernels.
 *
 * @see OperationKernels.cuh
 *
 * @author Mgepahmge
 * @date 2024/12/07
 */
namespace nz::krnl {
    /**
     * @brief Kernel function to perform matrix addition on GPU
     *
     * This function is designed to execute matrix addition using CUDA technology,
     * leveraging parallel computing capabilities of the GPU for efficient processing
     * of large datasets. It takes two input arrays of floats and stores their sum
     * in a third array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param a Pointer to the first input matrix elements stored as a one-dimensional array
     * @param b Pointer to the second input matrix elements stored as a one-dimensional array
     * @param c Pointer to the output matrix where the result will be stored, allocated by the caller
     * @param n The size of the matrix, representing the number of elements along one dimension (for a square matrix, total elements are n*n)
     */
    void MatrixAdd(dim3 gridDim, dim3 blockDim, const float* a, const float* b, float* c, const unsigned long long n);

    /**
     * @brief Kernel function to perform matrix subtraction on GPU
     *
     * This function is designed to execute matrix subtraction using CUDA technology,
     * leveraging parallel computing capabilities of the GPU for efficient processing
     * of large datasets. It takes two input arrays of floats and stores their difference
     * in a third array.
     *
     * @param a Pointer to the first input matrix elements stored as a one-dimensional array
     * @param b Pointer to the second input matrix elements stored as a one-dimensional array
     * @param c Pointer to the output matrix where the result will be stored, allocated by the caller
     * @param n The size of the matrix, representing the number of elements along one dimension (for a square matrix, total elements are n*n)
     */
    void MatrixSub(const dim3 gridDim, const dim3 blockDim, const float* a, const float* b, float* c,
                   const unsigned long long n);

    /**
     * @brief Kernel function to perform general matrix multiplication on GPU
     *
     * This function is designed to execute general matrix multiplication using CUDA technology,
     * leveraging parallel computing capabilities of the GPU for efficient processing
     * of large datasets. It takes two input arrays of floats and stores their product
     * in a third array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A Pointer to the first input matrix elements stored as a one-dimensional array
     * @param B Pointer to the second input matrix elements stored as a one-dimensional array
     * @param C Pointer to the output matrix where the result will be stored, allocated by the caller
     * @param M The number of rows in the first input matrix
     * @param N The number of columns in the second input matrix;
     * @param K The number of columns in the first input matrix
     */
    void GeneralMatrixMul(const dim3 gridDim, const dim3 blockDim, const float* A, const float* B, float* C,
                          const unsigned long long M,
                          const unsigned long long N,
                          const unsigned long long K);

    /**
     * @brief Kernel function to transpose a matrix on the GPU
     *
     * This function performs the transposition of a matrix on the GPU, swapping rows and columns.
     * The resulting transposed matrix is stored in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param d_A Pointer to the input matrix elements stored as a one-dimensional array
     * @param d_B Pointer to the output matrix where the transposed result will be stored
     * @param rows The number of rows in the input matrix
     * @param cols The number of columns in the input matrix
     */
    void Transpose(const dim3 gridDim, const dim3 blockDim, const float* d_A, float* d_B,
                   const unsigned int rows,
                   const unsigned int cols);

    /**
     * @brief Kernel function to perform scalar multiplication on the GPU
     *
     * This function multiplies each element of the input array by a scalar value and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the result will be stored
     * @param in Pointer to the input array elements
     * @param num The scalar value to multiply each element of the input array by
     * @param n The number of elements in the input and output arrays
     */
    void ScalarMul(const dim3 gridDim, const dim3 blockDim, float* out, const float* in, const float num,
                   const unsigned long long n);

    /**
     * @brief Kernel function to perform scalar division on the GPU
     *
     * This function divides each element of the input array by a scalar value and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the result will be stored
     * @param in Pointer to the input array elements
     * @param num The scalar value to divide each element of the input array by
     * @param n The number of elements in the input and output arrays
     */
    void ScalarDiv(const dim3 gridDim, const dim3 blockDim, float* out, const float* in, const float num,
                   const unsigned long long n);

    /**
     * @brief Kernel function to add a scalar to each element of a matrix on the GPU
     *
     * This function adds a scalar value to each element of the input array and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the result will be stored
     * @param in Pointer to the input array elements
     * @param num The scalar value to add to each element of the input array
     * @param n The number of elements in the input and output arrays
     */
    void ScalarAdd(const dim3 gridDim, const dim3 blockDim, float* out, const float* in, const float num,
                   const unsigned long long n);

    /**
     * @brief Kernel function to negate each element of a matrix on the GPU
     *
     * This function negates each element of the input array and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the negated result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     */
    void Negation(const dim3 gridDim, const dim3 blockDim, float* out, const float* in, const unsigned long long n);

    /**
     * @brief Kernel function to compute the reciprocal of each element of a matrix on the GPU
     *
     * This function computes the reciprocal (1/x) of each element of the input array and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the reciprocal result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     */
    void Recip(const dim3 gridDim, const dim3 blockDim, float* out, const float* in, const unsigned long long n);

    /**
     * @brief Kernel function to apply the Rectified Linear Unit (ReLU) activation on the GPU
     *
     * This function applies the ReLU activation function (max(0, x)) to each element of the input array and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the ReLU result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     */
    void RectifiedLinearUnit(const dim3 gridDim, const dim3 blockDim, float* out, const float* in,
                             const unsigned long long n);

    /**
     * @brief Kernel function to compute the gradient of the ReLU activation during backpropagation
     *
     * This function computes the gradient of the ReLU activation function during backpropagation
     * (dL/dx = dL/dy * (x > 0)) and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A_grad Pointer to the output array where the gradient result will be stored
     * @param A Pointer to the input array elements (before activation)
     * @param B_grad Pointer to the gradient of the next layer
     * @param n The number of elements in the arrays
     */
    void ReLUBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, const float* A, const float* B_grad,
                      const unsigned long long n);

    /**
     * @brief Kernel function to apply the Sigmoid activation function on the GPU
     *
     * This function applies the Sigmoid activation function (1 / (1 + exp(-x))) to each element of the input array
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the Sigmoid result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     */
    void Sigmoid(const dim3 gridDim, const dim3 blockDim, float* out, const float* in,
                  const unsigned long long n);

    /**
     * @brief Kernel function to compute the gradient of the Sigmoid activation during backpropagation
     *
     * This function computes the gradient of the Sigmoid activation function during backpropagation
     * (dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))) and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A_grad Pointer to the output array where the gradient result will be stored
     * @param B Pointer to the input array elements (after activation)
     * @param B_grad Pointer to the gradient of the next layer
     * @param n The number of elements in the arrays
     */
    void SigmoidBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, const float* B, const float* B_grad,
                         const unsigned long long n);

    /**
     * @brief Kernel function to apply the Tanh activation function on the GPU
     *
     * This function applies the Tanh activation function (tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)))
     * to each element of the input array and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the Tanh result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     */
    void Tanh(const dim3 gridDim, const dim3 blockDim, float* out, const float* in,
               const unsigned long long n);

    /**
     * @brief Kernel function to compute the gradient of the Tanh activation during backpropagation
     *
     * This function computes the gradient of the Tanh activation function during backpropagation
     * (dL/dx = dL/dy * (1 - tanh(x)^2)) and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A_grad Pointer to the output array where the gradient result will be stored
     * @param B Pointer to the input array elements (after activation)
     * @param B_grad Pointer to the gradient of the next layer
     * @param n The number of elements in the arrays
     */
    void TanhBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, const float* B, const float* B_grad,
                      const unsigned long long n);

    /**
     * @brief Kernel function to apply the Leaky ReLU activation function on the GPU
     *
     * This function applies the Leaky ReLU activation function (max(alpha * x, x)) to each element of the input array
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the Leaky ReLU result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     * @param alpha The slope of the negative part of the Leaky ReLU (default 0.01)
     */
    void LeakyReLU(const dim3 gridDim, const dim3 blockDim, float* out, const float* in,
                    const unsigned long long n, const float alpha = 0.01f);

    /**
     * @brief Kernel function to compute the gradient of the Leaky ReLU activation during backpropagation
     *
     * This function computes the gradient of the Leaky ReLU activation function during backpropagation
     * (dL/dx = dL/dy * (x > 0 ? 1 : alpha)) and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A_grad Pointer to the output array where the gradient result will be stored
     * @param A Pointer to the input array elements (before activation)
     * @param B_grad Pointer to the gradient of the next layer
     * @param n The number of elements in the arrays
     * @param alpha The slope of the negative part of the Leaky ReLU (default 0.01)
     */
    void LeakyReLUBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                      float alpha = 0.01f);

    /**
     * @brief Kernel function to apply the Swish activation function on the GPU
     *
     * This function applies the Swish activation function (x * sigmoid(x)) to each element of the input array
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the Swish result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     */
    void Swish(const dim3 gridDim, const dim3 blockDim, float* out, const float* in,
                 const unsigned long long n);

    /**
     * @brief Kernel function to compute the gradient of the Swish activation during backpropagation
     *
     * This function computes the gradient of the Swish activation function during backpropagation
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A_grad Pointer to the output array where the gradient result will be stored
     * @param A Pointer to the input array elements (before activation)
     * @param B Pointer to the output array elements (after activation)
     * @param B_grad Pointer to the gradient of the next layer
     * @param n The number of elements in the arrays
     */
    void SwishBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, const float* A, const float* B,
                         const float* B_grad, const unsigned long long n);

    /**
     * @brief Kernel function to apply the Exponential Linear Unit (ELU) activation function on the GPU
     *
     * This function applies the ELU activation function (x if x > 0, alpha * (exp(x) - 1) if x <= 0)
     * to each element of the input array and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the ELU result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     * @param alpha The alpha parameter used for negative values (default 1.0)
     */
    void ExponentialLinearUnit(const dim3 gridDim, const dim3 blockDim, float* out, const float* in, unsigned long long n, float alpha = 1.0f);

    /**
     * @brief Kernel function to compute the gradient of the ELU activation during backpropagation
     *
     * This function computes the gradient of the ELU activation function during backpropagation
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A_grad Pointer to the output array where the gradient result will be stored
     * @param A Pointer to the input array elements (before activation)
     * @param B_grad Pointer to the gradient of the next layer
     * @param n The number of elements in the arrays
     * @param alpha The alpha parameter used for negative values (default 1.0)
     */
    void ELUBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                float alpha = 1.0f);

    /**
     * @brief Kernel function to apply the Hard Sigmoid activation function on the GPU
     *
     * This function applies the Hard Sigmoid activation function (min(max(alpha * x + beta, 0), 1))
     * to each element of the input array and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the Hard Sigmoid result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     * @param alpha The slope of the Hard Sigmoid (default 0.2)
     * @param beta The offset of the Hard Sigmoid (default 0.5)
     */
    void HardSigmoid(const dim3 gridDim, const dim3 blockDim, float* out, const float* in,
                      const unsigned long long n, const float alpha = 0.2f, const float beta = 0.5f);

    /**
     * @brief Kernel function to compute the gradient of the Hard Sigmoid activation during backpropagation
     *
     * This function computes the gradient of the Hard Sigmoid activation function during backpropagation
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A_grad Pointer to the output array where the gradient result will be stored
     * @param A Pointer to the input array elements (before activation)
     * @param B_grad Pointer to the gradient of the next layer
     * @param n The number of elements in the arrays
     * @param alpha The slope of the Hard Sigmoid (default 0.2)
     * @param beta The offset of the Hard Sigmoid (default 0.5)
     */
    void HardSigmoidBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                        float alpha = 0.2f, float beta = 0.5f);

    /**
     * @brief Kernel function to apply the Hard Swish activation function on the GPU
     *
     * This function applies the Hard Swish activation function (x * HardSigmoid(x))
     * to each element of the input array and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the Hard Swish result will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input and output arrays
     * @param alpha The slope of the Hard Sigmoid (default 0.2)
     * @param beta The offset of the Hard Sigmoid (default 0.5)
     */
    void HardSwish(const dim3 gridDim, const dim3 blockDim, float* out, const float* in, unsigned long long n, float alpha = 0.2f, float beta = 0.5f);

    /**
     * @brief Kernel function to compute the gradient of the Hard Swish activation during backpropagation
     *
     * This function computes the gradient of the Hard Swish activation function during backpropagation
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A_grad Pointer to the output array where the gradient result will be stored
     * @param A Pointer to the input array elements (before activation)
     * @param B_grad Pointer to the gradient of the next layer
     * @param n The number of elements in the arrays
     * @param alpha The slope of the Hard Sigmoid (default 0.2)
     * @param beta The offset of the Hard Sigmoid (default 0.5)
     */
    void HardSwishBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, const float* A, const float* B_grad, unsigned long long n,
                                      float alpha = 0.2f, float beta = 0.5f);

    /**
     * @brief Kernel function to compute the summation of exponentials of each element in the input array
     *
     * This function computes the summation of exponentials of all elements in the input array
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param sharedMemSize The size of the shared memory buffer used by the kernel
     * @param out Pointer to the output array where the summation of exponentials will be stored
     * @param g_data Pointer to the input array elements
     * @param n The number of elements in the input array
     */
    void SummationExp(const dim3 gridDim, const dim3 blockDim, const size_t sharedMemSize, float* out, const float* g_data,
                                 const unsigned long long n);

    /**
     * @brief Kernel function to apply the Softmax function on the GPU
     *
     * This function applies the Softmax activation function, which normalizes the input values
     * by exponentiating them and dividing by the sum of all exponentials, to each element of the input array
     * and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the Softmax result will be stored
     * @param in Pointer to the input array elements
     * @param exp_sum_of_input The sum of the exponentials of the input array elements
     * @param n The number of elements in the input and output arrays
     */
    void Softmax(const dim3 gridDim, const dim3 blockDim, float* out, const float* in, const float exp_sum_of_input,
                 const unsigned long long n);

    /**
     * @brief Kernel function to compute the Jacobian of the Softmax function
     *
     * This function computes the Jacobian matrix of the Softmax function and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the Jacobian matrix will be stored
     * @param in Pointer to the input array elements
     * @param n The number of elements in the input array
     */
    void SoftmaxJacobian(const dim3 gridDim, const dim3 blockDim, float* out, const float* in,
                         const unsigned long long n);

    /**
     * @brief Kernel function to compute the Mean Squared Error (MSE) loss between predicted and real values
     *
     * This function computes the Mean Squared Error loss between the predicted and real values
     * for each element in the input arrays and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param sharedMemSize The size of the shared memory buffer used by the kernel
     * @param out Pointer to the output array where the MSE result will be stored
     * @param predict Pointer to the predicted values
     * @param real Pointer to the real values
     * @param n The number of elements in the input arrays
     */
    void MeanSquaredError(const dim3 gridDim, const dim3 blockDim, const size_t sharedMemSize, float* out,
                          const float* predict, const float* real, const unsigned long long n);

    /**
     * @brief Kernel function to compute the gradient of the Mean Squared Error (MSE) loss for backpropagation
     *
     * This function computes the gradient of the Mean Squared Error loss between the predicted and real values
     * for each element in the input arrays and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the MSE gradient will be stored
     * @param predict Pointer to the predicted values
     * @param real Pointer to the real values
     * @param n The number of elements in the input arrays
     */
    void MSEBackward(const dim3 gridDim, const dim3 blockDim, float* out, const float* predict,
                     const float* real, const unsigned long long n);

    /**
     * @brief Kernel function to perform Stochastic Gradient Descent (SGD) optimization
     *
     * This function updates the data array by applying Stochastic Gradient Descent with the given learning rate
     * and gradient for each element in the input arrays.
     *
     * @param data Pointer to the data array that will be updated
     * @param grad Pointer to the gradient array
     * @param lr The learning rate used for the gradient update
     * @param n The number of elements in the data and gradient arrays
     */
    void StochasticGradientDescent(const dim3 gridDim, const dim3 blockDim, float* data, const float* grad,
                                   const float lr, const unsigned long long n);

    /**
     * @brief Kernel function to compute the Binary Cross Entropy (BCE) loss between predicted and real values
     *
     * This function computes the Binary Cross Entropy loss between the predicted and real values
     * for each element in the input arrays and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param sharedMemSize The size of the shared memory buffer used by the kernel
     * @param out Pointer to the output array where the BCE result will be stored
     * @param predict Pointer to the predicted values
     * @param real Pointer to the real values
     * @param n The number of elements in the input arrays
     */
    void BinaryCrossEntropy(const dim3 gridDim, const dim3 blockDim, const size_t sharedMemSize, float* out,
                            const float* predict, const float* real, const unsigned long long n);

    /**
     * @brief Kernel function to compute the gradient of Binary Cross Entropy (BCE) loss for backpropagation
     *
     * This function computes the gradient of the Binary Cross Entropy loss between the predicted and real values
     * for each element in the input arrays and stores the result in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array where the BCE gradient will be stored
     * @param predict Pointer to the predicted values
     * @param real Pointer to the real values
     * @param n The number of elements in the input arrays
     */
    void BCEBackward(const dim3 gridDim, const dim3 blockDim, float* out, const float* predict,
                     const float* real, const unsigned long long n);

    /**
     * @brief Kernel function to apply Momentum optimization
     *
     * This function updates the output array using the Momentum optimization method, which incorporates the
     * previous velocity to smooth the gradient update.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param output Pointer to the output array that will be updated
     * @param grad Pointer to the gradient array
     * @param velocity Pointer to the previous velocity array
     * @param beta The momentum factor (typically between 0.9 and 0.99)
     * @param n The number of elements in the output, gradient, and velocity arrays
     */
    void Momentum(dim3 gridDim, dim3 blockDim, float* output, const float* grad, const float* velocity, float beta, unsigned long long n);

    /**
     * @brief Kernel function to apply AdaGrad optimization
     *
     * This function updates the data array using AdaGrad optimization, adjusting the learning rate for each
     * parameter based on the historical gradient squared values.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param data Pointer to the data array that will be updated
     * @param G Pointer to the array of accumulated squared gradients
     * @param grad Pointer to the gradient array
     * @param lr The learning rate used for the gradient update
     * @param eps A small constant to avoid division by zero (default 1e-8)
     * @param n The number of elements in the data, gradient, and accumulated gradient arrays
     */
    void AdaGrad(const dim3 gridDim, const dim3 blockDim, float* data, float* G, const float* grad, const float lr,
                 const float eps, const unsigned long long n);

    /**
     * @brief Kernel function to apply RMSprop optimization
     *
     * This function updates the data array using RMSprop optimization, which divides the gradient by the moving
     * average of the squared gradient values.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param data Pointer to the data array that will be updated
     * @param v Pointer to the array of accumulated squared gradients
     * @param grad Pointer to the gradient array
     * @param lr The learning rate used for the gradient update
     * @param beta The smoothing factor (typically between 0.9 and 0.99)
     * @param eps A small constant to avoid division by zero (default 1e-8)
     * @param n The number of elements in the data, gradient, and accumulated squared gradient arrays
     */
    void RMSprop(const dim3 gridDim, const dim3 blockDim, float* data, float* v, const float* grad, const float lr,
                 const float beta, const float eps, const unsigned long long n);

    /**
     * @brief Kernel function to apply Adam optimization
     *
     * This function updates the data array using Adam optimization, which combines momentum and RMSprop
     * to adaptively adjust the learning rates of each parameter.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param data Pointer to the data array that will be updated
     * @param m Pointer to the first moment estimate (mean of gradients)
     * @param v Pointer to the second moment estimate (variance of gradients)
     * @param grad Pointer to the gradient array
     * @param lr The learning rate used for the gradient update
     * @param beta1 The exponential decay rate for the first moment estimate (default 0.9)
     * @param beta2 The exponential decay rate for the second moment estimate (default 0.999)
     * @param eps A small constant to avoid division by zero (default 1e-8)
     * @param t The current time step or iteration
     * @param n The number of elements in the data, gradient, and moment arrays
     */
    void Adam(const dim3 gridDim, const dim3 blockDim, float* data, float* m, float* v, const float* grad,
              const float lr, const float beta1, const float beta2, const float eps, const int t,
              const unsigned long long n);

    /**
     * @brief Kernel function to apply NAdam optimization
     *
     * This function updates the data array using NAdam optimization, which combines Adam with Nesterov momentum.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param data Pointer to the data array that will be updated
     * @param m Pointer to the first moment estimate (mean of gradients)
     * @param m_modified Pointer to the modified first moment estimate for Nesterov momentum
     * @param v Pointer to the second moment estimate (variance of gradients)
     * @param grad Pointer to the gradient array
     * @param lr The learning rate used for the gradient update
     * @param beta1 The exponential decay rate for the first moment estimate (default 0.9)
     * @param beta2 The exponential decay rate for the second moment estimate (default 0.999)
     * @param eps A small constant to avoid division by zero (default 1e-8)
     * @param t The current time step or iteration
     * @param n The number of elements in the data, gradient, and moment arrays
     */
    void NAdam(const dim3 gridDim, const dim3 blockDim, float* data, float* m, float* m_modified, float* v,
               const float* grad, const float lr, const float beta1, const float beta2, const float eps, const int t,
               const unsigned long long n);

    /**
     * @brief Kernel function to apply AdaDelta optimization
     *
     * This function updates the data array using AdaDelta optimization, which uses a moving average of squared gradients
     * and deltas to adaptively adjust the learning rate.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param data Pointer to the data array that will be updated
     * @param acc_delta Pointer to the accumulated delta values
     * @param acc_grad Pointer to the accumulated gradient squared values
     * @param grad Pointer to the gradient array
     * @param rho The decay rate for the moving averages (typically between 0.9 and 0.95)
     * @param eps A small constant to avoid division by zero (default 1e-8)
     * @param n The number of elements in the data, gradient, and accumulated values arrays
     */
    void AdaDelta(const dim3 gridDim, const dim3 blockDim, float* data, float* acc_delta, float* acc_grad,
                  const float* grad, const float rho, const float eps, const unsigned long long n);

}

#endif
#endif //OPERATIONKERNELS_CUH

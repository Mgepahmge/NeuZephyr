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
#include <vector>
#include "Dimension.cuh"

#define OUTPUT_DIM(INPUT, KERNEL, STRIDE, PADDING) \
( ((size_t)(INPUT) + 2*(size_t)(PADDING) - (size_t)(KERNEL)) / (size_t)(STRIDE) + 1 )

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
#ifdef __CUDACC__
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
     * @param offset_c
     * @param offset_a
     * @param offset_b
     */
    void MatrixAdd(dim3 gridDim, dim3 blockDim, float* a, float* b, float* c, unsigned long long n, size_t offset_c = 0,
                   size_t offset_a = 0, size_t offset_b = 0);

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
     * @param offset_c
     * @param offset_a
     * @param offset_b
     */
    void MatrixAdd(dim3 gridDim, dim3 blockDim, float* a, float* b, float* c,
                   unsigned long long n, const std::vector<size_t>& offset_c, const std::vector<size_t>& offset_a,
                   const std::vector<size_t>& offset_b);

    /**
     * @brief Kernel function to perform matrix subtraction on GPU
     *
     * This function is designed to execute matrix subtraction using CUDA technology,
     * leveraging parallel computing capabilities of the GPU for efficient processing
     * of large datasets. It takes two input arrays of floats and stores their difference
     * in a third array.
     *
     * @param gridDim
     * @param blockDim
     * @param a Pointer to the first input matrix elements stored as a one-dimensional array
     * @param b Pointer to the second input matrix elements stored as a one-dimensional array
     * @param c Pointer to the output matrix where the result will be stored, allocated by the caller
     * @param n The size of the matrix, representing the number of elements along one dimension (for a square matrix, total elements are n*n)
     * @param offset_c
     * @param offset_a
     * @param offset_b
     */
    void MatrixSub(dim3 gridDim, dim3 blockDim, float* a, float* b, float* c,
                   unsigned long long n, size_t offset_c = 0, size_t offset_a = 0, size_t offset_b = 0);

    /**
     * @brief Kernel function to perform matrix subtraction on GPU
     *
     * This function is designed to execute matrix subtraction using CUDA technology,
     * leveraging parallel computing capabilities of the GPU for efficient processing
     * of large datasets. It takes two input arrays of floats and stores their difference
     * in a third array.
     *
     * @param gridDim
     * @param blockDim
     * @param a Pointer to the first input matrix elements stored as a one-dimensional array
     * @param b Pointer to the second input matrix elements stored as a one-dimensional array
     * @param c Pointer to the output matrix where the result will be stored, allocated by the caller
     * @param n The size of the matrix, representing the number of elements along one dimension (for a square matrix, total elements are n*n)
     * @param offset_c
     * @param offset_a
     * @param offset_b
     */
    void MatrixSub(dim3 gridDim, dim3 blockDim, float* a, float* b, float* c,
                   unsigned long long n, const std::vector<size_t>& offset_c, const std::vector<size_t>& offset_a,
                   const std::vector<size_t>& offset_b);

    /**
     * @brief Kernel function to perform single-precision matrix multiplication on GPU using CUDA cores
     *
     * This function is designed to execute general matrix multiplication using CUDA technology,
     * leveraging the parallel computing capabilities of the GPU for efficient processing
     * of large datasets. It performs single-precision (FP32) matrix multiplication on the CUDA cores,
     * taking two input arrays of floats and storing their product in a third array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A Pointer to the first input matrix elements stored as a one-dimensional array
     * @param B Pointer to the second input matrix elements stored as a one-dimensional array
     * @param C Pointer to the output matrix where the result will be stored, allocated by the caller
     * @param M The number of rows in matrix A and matrix C
     * @param N The number of columns in matrix B and matrix C
     * @param K The number of columns in matrix A and rows in matrix B
     * @param offset_c
     * @param offset_a
     * @param offset_b
     */
    void GeneralMatrixMul(dim3 gridDim, dim3 blockDim, float* A, float* B, float* C,
                          unsigned long long M,
                          unsigned long long N,
                          unsigned long long K,
                          size_t offset_c = 0,
                          size_t offset_a = 0,
                          size_t offset_b = 0);

    /**
     * @brief Kernel function to perform single-precision matrix multiplication on GPU using CUDA cores
     *
     * This function is designed to execute general matrix multiplication using CUDA technology,
     * leveraging the parallel computing capabilities of the GPU for efficient processing
     * of large datasets. It performs single-precision (FP32) matrix multiplication on the CUDA cores,
     * taking two input arrays of floats and storing their product in a third array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param A Pointer to the first input matrix elements stored as a one-dimensional array
     * @param B Pointer to the second input matrix elements stored as a one-dimensional array
     * @param C Pointer to the output matrix where the result will be stored, allocated by the caller
     * @param M The number of rows in matrix A and matrix C
     * @param N The number of columns in matrix B and matrix C
     * @param K The number of columns in matrix A and rows in matrix B
     * @param offset_c
     * @param offset_a
     * @param offset_b
     */
    void GeneralMatrixMul(dim3 gridDim, dim3 blockDim, float* A, float* B, float* C,
                          unsigned long long M,
                          unsigned long long N,
                          unsigned long long K,
                          const std::vector<size_t>& offset_c,
                          const std::vector<size_t>& offset_a,
                          const std::vector<size_t>& offset_b);

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
     * @param offset The offset within the input and output arrays
     */
    void Transpose(dim3 gridDim, dim3 blockDim, float* d_A, float* d_B,
                   unsigned int rows,
                   unsigned int cols,
                   size_t offset = 0);

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
     * @param offset The offset within the input and output arrays
     */
    void Transpose(dim3 gridDim, dim3 blockDim, float* d_A, float* d_B,
                   unsigned int rows,
                   unsigned int cols,
                   const std::vector<size_t>& offset);

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
    void ScalarMul(dim3 gridDim, dim3 blockDim, float* out, float* in, float num,
                   unsigned long long n);

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
    void ScalarDiv(dim3 gridDim, dim3 blockDim, float* out, float* in, float num,
                   unsigned long long n);

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
    void ScalarAdd(dim3 gridDim, dim3 blockDim, float* out, float* in, float num,
                   unsigned long long n);

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
    void Negation(dim3 gridDim, dim3 blockDim, float* out, float* in, unsigned long long n);

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
    void Recip(dim3 gridDim, dim3 blockDim, float* out, float* in, unsigned long long n);

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
    void RectifiedLinearUnit(dim3 gridDim, dim3 blockDim, float* out, float* in,
                             unsigned long long n);

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
    void ReLUBackward(dim3 gridDim, dim3 blockDim, float* A_grad, float* A, float* B_grad,
                      unsigned long long n);

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
    void Sigmoid(dim3 gridDim, dim3 blockDim, float* out, float* in,
                 unsigned long long n);

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
    void SigmoidBackward(dim3 gridDim, dim3 blockDim, float* A_grad, float* B, float* B_grad,
                         unsigned long long n);

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
    void Tanh(dim3 gridDim, dim3 blockDim, float* out, float* in,
              unsigned long long n);

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
    void TanhBackward(dim3 gridDim, dim3 blockDim, float* A_grad, float* B, float* B_grad,
                      unsigned long long n);

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
    void LeakyReLU(dim3 gridDim, dim3 blockDim, float* out, float* in,
                   unsigned long long n, float alpha = 0.01f);

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
    void LeakyReLUBackward(dim3 gridDim, dim3 blockDim, float* A_grad, float* A, float* B_grad,
                           unsigned long long n,
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
    void Swish(dim3 gridDim, dim3 blockDim, float* out, float* in,
               unsigned long long n);

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
    void SwishBackward(dim3 gridDim, dim3 blockDim, float* A_grad, float* A, float* B,
                       float* B_grad, unsigned long long n);

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
    void ExponentialLinearUnit(dim3 gridDim, dim3 blockDim, float* out, float* in,
                               unsigned long long n, float alpha = 1.0f);

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
    void ELUBackward(dim3 gridDim, dim3 blockDim, float* A_grad, float* A, float* B_grad,
                     unsigned long long n,
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
    void HardSigmoid(dim3 gridDim, dim3 blockDim, float* out, float* in,
                     unsigned long long n, float alpha = 0.2f, float beta = 0.5f);

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
    void HardSigmoidBackward(dim3 gridDim, dim3 blockDim, float* A_grad, float* A,
                             float* B_grad, unsigned long long n,
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
    void HardSwish(dim3 gridDim, dim3 blockDim, float* out, float* in, unsigned long long n,
                   float alpha = 0.2f, float beta = 0.5f);

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
    void HardSwishBackward(dim3 gridDim, dim3 blockDim, float* A_grad, float* A, float* B_grad,
                           unsigned long long n,
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
     * @param offset
     */
    void SummationExp(dim3 gridDim, dim3 blockDim, size_t sharedMemSize, float* out,
                      float* g_data,
                      unsigned long long n, size_t offset = 0);

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
     * @param offset
     */
    void Softmax(dim3 gridDim, dim3 blockDim, float* out, float* in, float exp_sum_of_input,
                 unsigned long long n, size_t offset = 0);

    void Softmax(dim3 gridDim, dim3 blockDim, float* out, float* in, const std::vector<float>& exp_sum_of_input,
                 unsigned long long n, const std::vector<size_t>& offset);

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
    void SoftmaxJacobian(dim3 gridDim, dim3 blockDim, float* out, float* in,
                         unsigned long long n);

    void SoftmaxJacobian(dim3 gridDim, dim3 blockDim, float* out, float* in,
                         unsigned long long n, const std::vector<size_t>& offset_o,
                         const std::vector<size_t>& offset_i);

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
    void MeanSquaredError(dim3 gridDim, dim3 blockDim, size_t sharedMemSize, float* out,
                          float* predict, float* real, unsigned long long n);

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
    void MSEBackward(dim3 gridDim, dim3 blockDim, float* out, float* predict,
                     float* real, unsigned long long n);

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
    void StochasticGradientDescent(dim3 gridDim, dim3 blockDim, float* data, float* grad,
                                   float lr, unsigned long long n);

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
    void BinaryCrossEntropy(dim3 gridDim, dim3 blockDim, size_t sharedMemSize, float* out,
                            float* predict, float* real, unsigned long long n);

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
    void BCEBackward(dim3 gridDim, dim3 blockDim, float* out, float* predict,
                     float* real, unsigned long long n);

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
    void Momentum(dim3 gridDim, dim3 blockDim, float* output, float* grad, float* velocity, float beta,
                  unsigned long long n);

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
    void AdaGrad(dim3 gridDim, dim3 blockDim, float* data, float* G, float* grad, float lr,
                 float eps, unsigned long long n);

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
    void RMSprop(dim3 gridDim, dim3 blockDim, float* data, float* v, float* grad, float lr,
                 float beta, float eps, unsigned long long n);

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
    void Adam(dim3 gridDim, dim3 blockDim, float* data, float* m, float* v, float* grad,
              float lr, float beta1, float beta2, float eps, int t,
              unsigned long long n);

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
    void NAdam(dim3 gridDim, dim3 blockDim, float* data, float* m, float* m_modified, float* v,
               float* grad, float lr, float beta1, float beta2, float eps, int t,
               unsigned long long n);

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
    void AdaDelta(dim3 gridDim, dim3 blockDim, float* data, float* acc_delta, float* acc_grad,
                  float* grad, float rho, float eps, unsigned long long n);

    /**
     * @brief Kernel function to perform fast matrix multiplication using Tensor Cores with half-precision (FP16) support
     *
     * This function performs matrix multiplication on two input matrices A and B using Tensor Cores, which are specialized
     * hardware units in modern GPUs designed for high-throughput matrix operations. The matrices are internally padded to
     * be multiples of 16 for efficient computation and then cropped back to their original dimensions after the operation.
     *
     * @param A Pointer to the first input matrix (of size M x K)
     * @param B Pointer to the second input matrix (of size K x N)
     * @param C Pointer to the result matrix (of size M x N)
     * @param M The number of rows in matrix A and matrix C
     * @param N The number of columns in matrix B and matrix C
     * @param K The number of columns in matrix A and rows in matrix B
     *
     * @note The matrices A and B are assumed to be padded to the nearest multiple of 16 for efficient computation.
     *       After the computation, the resulting matrix C will be cropped back to the original dimensions (M x N).
     */
    void TensorCoreGEMM(float* A, float* B, float* C, unsigned long long M,
                        unsigned long long N, unsigned long long K);

    void TensorCoreGEMMParallel(float* A, float* B, float* C,
                                const data::Dimension& A_shape,
                                const data::Dimension& B_shape,
                                const data::Dimension& C_shape);

    void GEMMBackwardParallel(float* A, float* B, float* C,
                              const data::Dimension& A_shape,
                              const data::Dimension& B_shape,
                              const data::Dimension& C_shape);

    /**
     * @brief Kernel function to fill a data array with a given value
     *
     * This function fills a data array with a specified value.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param data Pointer to the data array that will be filled
     * @param value The value to fill the array with
     * @param n The number of elements in the data array
     * @param offset
     *
     * @note This function is used for initializing the data array with a given value.
     */
    void Fill(dim3 gridDim, dim3 blockDim, float* data, float value, unsigned long long n, size_t offset = 0);

    /**
     * @brief Kernel function to perform element-wise Hadamard product of two arrays
     *
     * This function performs element-wise Hadamard product of two input arrays and stores the result in an output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array
     * @param in1 Pointer to the first input array
     * @param in2 Pointerto the second input array
     * @param n The number of elements in the arrays
     *
     * @note This function is used for computing the element-wise Hadamard product of two arrays.
     */
    void HadamardProduct(dim3 gridDim, dim3 blockDim, float* out, float* in1, float* in2,
                         unsigned long long n);

    /**
     * @brief Kernel function to perform element-wise division of two arrays
     *
     * This function performs element-wise division of two input arrays and stores the result in an output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param out Pointer to the output array
     * @param in1 Pointer to the first input array
     * @param in2 Pointerto the second input array
     * @param n The number of elements in the arrays
     * @param offset_o
     * @param offset_1
     * @param offset_2
     *
     * @note This function is used for computing the element-wise division of two arrays.
     */
    void ElementwiseDivide(dim3 gridDim, dim3 blockDim, float* out, float* in1, float* in2,
                           unsigned long long n, size_t offset_o = 0, size_t offset_1 = 0, size_t offset_2 = 0);

    void ElementwiseDivide(dim3 gridDim, dim3 blockDim, float* out, float* in1, float* in2,
                           unsigned long long n, const std::vector<size_t>& offset_o,
                           const std::vector<size_t>& offset_1,
                           const std::vector<size_t>& offset_2);

    /**
     * @brief Kernel function to perform element-wise summation of two arrays
     *
     * This function performs element-wise summation of two input arrays and stores the result in an output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration
     * @param blockDim The block dimensions for the CUDA kernel launch configuration
     * @param sharedMemSize The size of the shared memory buffer
     * @param out Pointer to the output array
     * @param in Pointer to the input array
     * @param n The number of elements in the arrays
     * @param offset
     *
     * @note This function is used for computing the element-wise summation of two arrays.
     */
    void Summation(dim3 gridDim, dim3 blockDim, unsigned long long sharedMemSize, float* out, float* in,
                   unsigned long long n, size_t offset = 0);

    /**
     * @brief Copies gradient data from one array to another with specified offsets.
     *
     * This kernel function performs a gradient copy operation, transferring data 
     * from the input array to the output array while applying offsets for both 
     * the input and output arrays.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the gradient data will be stored.
     * @param in Pointer to the input array containing the gradient data to be copied.
     * @param n The number of elements to copy.
     * @param offset_o A vector of offsets for the output array.
     * @param offset_i A vector of offsets for the input array.
     *
     * @note This function is designed for use in GPU-based gradient operations 
     *       and assumes that the input and output arrays are properly allocated 
     *       and accessible on the device.
     */
    void gradCopy(dim3 gridDim, dim3 blockDim, float* out, float* in, size_t n,
                  const std::vector<size_t>& offset_o, const std::vector<size_t>& offset_i);

    /**
     * @brief Copies gradient data from one array to another with specified offsets.
     *
     * This kernel function performs a gradient copy operation, transferring data
     * from the input array to the output array while applying offsets for both
     * the input and output arrays.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the gradient data will be stored.
     * @param in Pointer to the input array containing the gradient data to be copied.
     * @param n The number of elements to copy.
     * @param offset_o A vector of offsets for the output array.
     * @param offset_i A vector of offsets for the input array.
     *
     * @note This function is designed for use in GPU-based gradient operations
     *       and assumes that the input and output arrays are properly allocated
     *       and accessible on the device.
     */
    void NgradCopy(dim3 gridDim, dim3 blockDim, float* out, float* in, size_t n,
                   const std::vector<size_t>& offset_o, const std::vector<size_t>& offset_i);

    /**
     * @brief Expands the input array into the output array with a specified total size.
     *
     * This kernel function takes an input array and expands it into an output array
     * by repeating or padding elements to match the specified total size.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the expanded data will be stored.
     * @param in Pointer to the input array containing the original data.
     * @param n The number of elements in the input array.
     * @param total The total number of elements in the output array after expansion.
     *
     * @note This function assumes that the input and output arrays are properly allocated
     *       and accessible on the device.
     */
    void Expand(dim3 gridDim, dim3 blockDim, float* out, float* in, size_t n,
                size_t total);

    /**
     * @brief Compresses the input array into the output array with a specified total size.
     *
     * This kernel function reduces the size of the input array by compressing its elements
     * into the output array to match the specified total size.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the compressed data will be stored.
     * @param in Pointer to the input array containing the original data.
     * @param n The number of elements in the input array.
     * @param total The total number of elements in the output array after compression.
     *
     * @note This function assumes that the input and output arrays are properly allocated
     *       and accessible on the device.
     */
    void Compress(dim3 gridDim, dim3 blockDim, float* out, float* in, size_t n,
                  size_t total);

    /**
     * @brief Rearranges image data into column format for convolution operations.
     *
     * This kernel function transforms the input image data into a columnar format
     * (im2col) to facilitate efficient convolution operations. It extracts patches
     * from the input image based on the kernel size, stride, and padding, and stores
     * them in the output array.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the columnar data will be stored.
     * @param in Pointer to the input image data array.
     * @param H_out The height of the output feature map.
     * @param W_out The width of the output feature map.
     * @param C The number of input channels.
     * @param K_h The height of the convolution kernel.
     * @param K_w The width of the convolution kernel.
     * @param stride The stride of the convolution operation.
     * @param pad The padding applied to the input image.
     * @param H_in The height of the input image.
     * @param W_in The width of the input image.
     * @param batch The number of images in the batch.
     *
     * @note This function assumes that the input and output arrays are properly allocated
     *       and accessible on the device.
     */
    void img2col(dim3 gridDim, dim3 blockDim, float* out, float* in, size_t H_out,
                 size_t W_out, size_t C, size_t K_h, size_t K_w, size_t stride,
                 size_t pad, size_t H_in, size_t W_in, size_t batch);

    /**
     * @brief Rearranges columnar data back into image format for backpropagation in convolution operations.
     *
     * This kernel function performs the reverse operation of `img2col`, transforming
     * columnar data back into its original image format. It is used during the
     * backpropagation phase of convolutional neural networks to reconstruct the
     * gradient of the input image.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the reconstructed image data will be stored.
     * @param in Pointer to the input columnar data array.
     * @param H_out The height of the output feature map.
     * @param W_out The width of the output feature map.
     * @param C The number of input channels.
     * @param K_h The height of the convolution kernel.
     * @param K_w The width of the convolution kernel.
     * @param stride The stride of the convolution operation.
     * @param pad The padding applied to the input image.
     * @param H_in The height of the input image.
     * @param W_in The width of the input image.
     * @param batch The number of images in the batch.
     *
     * @note This function assumes that the input and output arrays are properly allocated
     *       and accessible on the device.
     */
    void img2colBackward(dim3 gridDim, dim3 blockDim, float* out, float* in, size_t H_out,
                         size_t W_out, size_t C, size_t K_h, size_t K_w, size_t stride,
                         size_t pad, size_t H_in, size_t W_in, size_t batch);

    /**
     * @brief Rearranges columnar data back into image format.
     *
     * This kernel function transforms columnar data into its original image format.
     * It is typically used in operations where data needs to be reconstructed
     * from a columnar representation, such as after convolution operations.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the reconstructed image data will be stored.
     * @param in Pointer to the input columnar data array.
     * @param H_out The height of the output image.
     * @param W_out The width of the output image.
     * @param C_out The number of output channels.
     * @param batches The number of images in the batch.
     *
     * @note This function assumes that the input and output arrays are properly allocated
     *       and accessible on the device.
     */
    void col2img(dim3 gridDim, dim3 blockDim, float* out, float* in, size_t H_out,
                 size_t W_out, size_t C_out, size_t batches);

    /**
     * @brief Rearranges columnar data back into image format for backpropagation.
     *
     * This kernel function transforms columnar data back into its original image format.
     * It is typically used during the backpropagation phase of convolutional neural networks
     * to reconstruct the gradient of the input image.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the reconstructed image data will be stored.
     * @param in Pointer to the input columnar data array.
     * @param H_out The height of the output image.
     * @param W_out The width of the output image.
     * @param C_out The number of output channels.
     * @param batches The number of images in the batch.
     *
     * @note This function assumes that the input and output arrays are properly allocated
     *       and accessible on the device.
     */
    void col2imgBackward(dim3 gridDim, dim3 blockDim, float* out, float* in, size_t H_out,
                         size_t W_out, size_t C_out, size_t batches);

    /**
     * @brief Kernel function to perform average pooling on the GPU.
     *
     * This function applies average pooling to the input tensor, reducing its spatial dimensions
     * by computing the average value within each pooling window.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the pooled results will be stored.
     * @param in Pointer to the input array containing the original data.
     * @param pool_size The size of the pooling window.
     * @param stride The stride of the pooling operation.
     * @param padding The padding applied to the input tensor.
     * @param batches The number of batches in the input tensor.
     * @param channels The number of channels in the input tensor.
     * @param H_in The height of the input tensor.
     * @param W_in The width of the input tensor.
     * @param H_out The height of the output tensor.
     * @param W_out The width of the output tensor.
     */
    void AveragePooling(dim3 gridDim, dim3 blockDim, float* out, float* in,
                        size_t pool_size, size_t stride, size_t padding,
                        size_t batches, size_t channels, size_t H_in, size_t W_in,
                        size_t H_out, size_t W_out);

    /**
     * @brief Kernel function to compute the gradient of average pooling during backpropagation.
     *
     * This function computes the gradient of the average pooling operation, distributing
     * the gradient values evenly across the pooling window.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param out Pointer to the output array where the gradient will be stored.
     * @param in Pointer to the input array containing the gradient from the next layer.
     * @param pool_size The size of the pooling window.
     * @param stride The stride of the pooling operation.
     * @param padding The padding applied to the input tensor.
     * @param batches The number of batches in the input tensor.
     * @param channels The number of channels in the input tensor.
     * @param H_in The height of the input tensor.
     * @param W_in The width of the input tensor.
     * @param H_out The height of the output tensor.
     * @param W_out The width of the output tensor.
     */
    void AveragePoolingBackward(dim3 gridDim, dim3 blockDim, float* out, float* in,
                                size_t pool_size, size_t stride, size_t padding,
                                size_t batches, size_t channels, size_t H_in, size_t W_in,
                                size_t H_out, size_t W_out);

    /**
     * @brief Kernel function to compute the gradient of global average pooling during backpropagation.
     *
     * This function computes the gradient of the global average pooling operation, distributing
     * the gradient values evenly across all spatial dimensions.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param output Pointer to the output array where the gradient will be stored.
     * @param in Pointer to the input array containing the gradient from the next layer.
     * @param batches The number of batches in the input tensor.
     * @param channels The number of channels in the input tensor.
     * @param height The height of the input tensor.
     * @param width The width of the input tensor.
     */
    void GlobalAvgPoolBackward(dim3 gridDim, dim3 blockDim, float* output, float* in,
                               size_t batches, size_t channels, size_t height, size_t width);

    /**
     * @brief Kernel function to perform max pooling on the GPU.
     *
     * This function applies max pooling to the input tensor, reducing its spatial dimensions
     * by selecting the maximum value within each pooling window.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param output Pointer to the output array where the pooled results will be stored.
     * @param position Pointer to the array where the positions of the maximum values will be stored.
     * @param input Pointer to the input array containing the original data.
     * @param pool_size The size of the pooling window.
     * @param stride The stride of the pooling operation.
     * @param padding The padding applied to the input tensor.
     * @param batches The number of batches in the input tensor.
     * @param channels The number of channels in the input tensor.
     * @param H_in The height of the input tensor.
     * @param W_in The width of the input tensor.
     * @param H_out The height of the output tensor.
     * @param W_out The width of the output tensor.
     */
    void MaxPooling(dim3 gridDim, dim3 blockDim, float* output, float* position, float* input,
                    size_t pool_size, size_t stride, size_t padding,
                    size_t batches, size_t channels, size_t H_in, size_t W_in,
                    size_t H_out, size_t W_out);

    /**
     * @brief Kernel function to compute the gradient of max pooling during backpropagation.
     *
     * This function computes the gradient of the max pooling operation, propagating
     * the gradient values only to the positions of the maximum values in the pooling window.
     *
     * @param gridDim The grid dimensions for the CUDA kernel launch configuration.
     * @param blockDim The block dimensions for the CUDA kernel launch configuration.
     * @param output Pointer to the output array where the gradient will be stored.
     * @param position Pointer to the array containing the positions of the maximum values.
     * @param input Pointer to the input array containing the gradient from the next layer.
     * @param pool_size The size of the pooling window.
     * @param stride The stride of the pooling operation.
     * @param padding The padding applied to the input tensor.
     * @param batches The number of batches in the input tensor.
     * @param channels The number of channels in the input tensor.
     * @param H_in The height of the input tensor.
     * @param W_in The width of the input tensor.
     * @param H_out The height of the output tensor.
     * @param W_out The width of the output tensor.
     */
    void MaxPoolingBackward(dim3 gridDim, dim3 blockDim, float* output, float* position, float* input,
                            size_t pool_size, size_t stride, size_t padding,
                            size_t batches, size_t channels, size_t H_in, size_t W_in,
                            size_t H_out, size_t W_out);
#endif
}

#endif //OPERATIONKERNELS_CUH

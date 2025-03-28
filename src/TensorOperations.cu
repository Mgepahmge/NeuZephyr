#include "NeuZephyr/TensorOperations.cuh"

#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"

namespace nz::data {
    void iRELU(float* output, float* input, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::RectifiedLinearUnit(grid, block, output, input, size);
    }

    void iSigmoid(float* output, float* input, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::Sigmoid(grid, block, output, input, size);
    }

    void iTanh(float* output, float* input, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::Tanh(grid, block, output, input, size);
    }

    void iLeakyReLU(float* output, float* input, const unsigned long long size, const float alpha) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::LeakyReLU(grid, block, output, input, size, alpha);
    }

    void iSwish(float* output, float* input, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::Swish(grid, block, output, input, size);
    }

    void iELU(float* output, float* input, const unsigned long long size, const float alpha) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::ExponentialLinearUnit(grid, block, output, input, size, alpha);
    }

    void iHardSigmoid(float* output, float* input, const unsigned long long size, const float alpha,
                      const float beta) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::HardSigmoid(grid, block, output, input, size, alpha, beta);
    }

    void iHardSwish(float* output, float* input, unsigned long long size, float alpha, float beta) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::HardSwish(grid, block, output, input, size, alpha, beta);
    }

    void iSoftmax(float* output, float* input, const float sum, const unsigned long long size, const size_t offset) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::Softmax(grid, block, output, input, sum, size, offset);
    }

    void iScalarAdd(float* output, float* input, const float scalar, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::ScalarAdd(grid, block, output, input, scalar, size);
    }

    void iScalarDiv(float* output, float* input, const float scalar, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::ScalarDiv(grid, block, output, input, scalar, size);
    }

    void iScalarMul(float* output, float* input, float scalar, unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::ScalarMul(grid, block, output, input, scalar, size);
    }
}

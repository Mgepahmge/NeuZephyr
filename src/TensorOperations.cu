#include "NeuZephyr/TensorOperations.cuh"

#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"

namespace nz::data {
    void iRELU(float* output, const float* input, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::RectifiedLinearUnit(grid, block, output, input, size);
        CHECK(cudaDeviceSynchronize());
    }

    void iSigmoid(float* output, const float* input, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::Sigmoid(grid, block, output, input, size);
        CHECK(cudaDeviceSynchronize());
    }

    void iTanh(float* output, const float* input, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::Tanh(grid, block, output, input, size);
        CHECK(cudaDeviceSynchronize());
    }

    void iLeakyReLU(float* output, const float* input, const unsigned long long size, const float alpha) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::LeakyReLU(grid, block, output, input, size, alpha);
        CHECK(cudaDeviceSynchronize());
    }

    void iSwish(float* output, const float* input, const unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::Swish(grid, block, output, input, size);
        CHECK(cudaDeviceSynchronize());
    }

    void iELU(float* output, const float* input, const unsigned long long size, const float alpha) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::ExponentialLinearUnit(grid, block, output, input, size, alpha);
        CHECK(cudaDeviceSynchronize());
    }

    void iHardSigmoid(float* output, const float* input, const unsigned long long size, const float alpha,
                      const float beta) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::HardSigmoid(grid, block, output, input, size, alpha, beta);
        CHECK(cudaDeviceSynchronize());
    }

    void iHardSwish(float* output, const float* input, unsigned long long size, float alpha, float beta) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::HardSwish(grid, block, output, input, size, alpha, beta);
        CHECK(cudaDeviceSynchronize());
    }

    void iSoftmax(float* output, const float* input, float sum, unsigned long long size) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((size + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::Softmax(grid, block, output, input, sum, size);
        CHECK(cudaDeviceSynchronize());
    }
}

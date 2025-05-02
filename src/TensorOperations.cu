#include "NeuZephyr/TensorOperations.cuh"

#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"
#include "NeuZephyr/utils.cuh"

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

    void iSoftmax(float* output, float* input, const std::vector<float>& sum, const unsigned long long size,
                  const std::vector<size_t>& offset) {
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

    void iMatrixAdd(float* out, float* in1, float* in2, const size_t n, const std::vector<size_t>& offset_o,
                    const std::vector<size_t>& offset_i1, const std::vector<size_t>& offset_i2) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((n + block.x - 1) / block.x);
        krnl::MatrixAdd(grid, block, in1, in2, out, n, offset_o, offset_i1, offset_i2);
    }

    void iMatrixSub(float* out, float* in1, float* in2, const size_t n, const std::vector<size_t>& offset_o,
                    const std::vector<size_t>& offset_i1, const std::vector<size_t>& offset_i2) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((n + block.x - 1) / block.x);
        krnl::MatrixSub(grid, block, in1, in2, out, n, offset_o, offset_i1, offset_i2);
    }

    void iElementwiseDivide(float* out, float* in1, float* in2, const size_t n, const std::vector<size_t>& offset_o,
                            const std::vector<size_t>& offset_i1, const std::vector<size_t>& offset_i2) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((n + block.x - 1) / block.x);
        krnl::ElementwiseDivide(grid, block, out, in1, in2, n, offset_o, offset_i1, offset_i2);
    }

    void iGeneralMatrixMul(float* A, float* B, float* C, const size_t M, const size_t N, const size_t K,
                           const std::vector<size_t>& offsetC, const std::vector<size_t>& offsetA,
                           const std::vector<size_t>& offsetB) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        krnl::GeneralMatrixMul(grid, block, A, B, C, M, N, K, offsetC, offsetA, offsetB);
    }

    void iTranspose(float* out, float* in, const size_t rows, const size_t cols, const std::vector<size_t>& offset) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
        krnl::Transpose(grid, block, in, out, rows, cols, offset);
    }
}

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

    void iTensorCoreGEMM(float* A, float* B, float* C, const Dimension& shapeA, const Dimension& shapeB,
        const Dimension& shapeC) {
        krnl::TensorCoreGEMMParallel(A, B, C, shapeA, shapeB, shapeC);
    }

    void iGEMMBackward(float* A, float* B, float* C, const Dimension& shapeA, const Dimension& shapeB,
        const Dimension& shapeC) {
        krnl::GEMMBackwardParallel(A, B, C, shapeA, shapeB, shapeC);
    }

    void iTranspose(float* out, float* in, const size_t rows, const size_t cols, const std::vector<size_t>& offset) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
        krnl::Transpose(grid, block, in, out, rows, cols, offset);
    }

    void iSoftmaxJacobian(float* out, float* in, const size_t n, const std::vector<size_t>& offset_o,
        const std::vector<size_t>& offset_i) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        krnl::SoftmaxJacobian(grid, block, out, in, n, offset_o, offset_i);
    }

    void iImg2col(float* out, float* in, const size_t H_out, const size_t W_out, const size_t C, const size_t K_h,
        const size_t K_w, const size_t stride, const size_t pad, const size_t H_in, const size_t W_in,
        const size_t batch) {
        const dim3 block(BLOCKSIZE);
        const dim3 grid((H_out * W_out * C * K_h * K_w * batch + BLOCKSIZE - 1) / BLOCKSIZE);
        krnl::img2col(grid, block, out, in, H_out, W_out, C, K_h, K_w, stride, pad, H_in, W_in, batch);
    }
}

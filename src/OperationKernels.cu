#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/utils.cuh"
#include "NeuZephyr/StreamManager.cuh"
#include "NeuZephyr/Dimension.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace nz::krnl {
    using namespace cuStrm;

    __global__ void MatrixAddKernel(float* c, const float* a, const float* b,
                                    const unsigned long long n,
                                    const size_t offset_c,
                                    const size_t offset_a,
                                    const size_t offset_b) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        float* c_m = c + offset_c;
        const float* a_m = a + offset_a;
        const float* b_m = b + offset_b;
        if (idx < n) {
            c_m[idx] = a_m[idx] + b_m[idx];
        }
    }

    void MatrixAdd(const dim3 gridDim, const dim3 blockDim, float* a, float* b, float* c,
                   const unsigned long long n, const size_t offset_c, const size_t offset_a, const size_t offset_b) {
        StreamManager<float>::Instance().submit(MatrixAddKernel, gridDim, blockDim, 0, c, a, b, n, offset_c, offset_a,
                                                offset_b);
    }

    void MatrixAdd(const dim3 gridDim, const dim3 blockDim, float* a, float* b, float* c,
                   const unsigned long long n, const std::vector<size_t>& offset_c, const std::vector<size_t>& offset_a,
                   const std::vector<size_t>& offset_b) {
        StreamManager<float>::Instance().submitParallel(MatrixAddKernel, gridDim, blockDim, 0, c, a, b, offset_c,
                                                        offset_a, offset_b, n);
    }

    __global__ void MatrixSubKernel(float* c, const float* a, const float* b,
                                    const unsigned long long n,
                                    const size_t offset_c,
                                    const size_t offset_a,
                                    const size_t offset_b) {
        unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx + offset_c] = a[idx + offset_a] - b[idx + offset_b];
        }
    }

    void MatrixSub(const dim3 gridDim, const dim3 blockDim, float* a, float* b, float* c,
                   const unsigned long long n, const size_t offset_c,
                   const size_t offset_a,
                   const size_t offset_b) {
        StreamManager<float>::Instance().submit(MatrixSubKernel, gridDim, blockDim, 0, c, a, b, n, offset_c, offset_a,
                                                offset_b);
    }

    void MatrixSub(const dim3 gridDim, const dim3 blockDim, float* a, float* b, float* c,
                   const unsigned long long n, const std::vector<size_t>& offset_c, const std::vector<size_t>& offset_a,
                   const std::vector<size_t>& offset_b) {
        StreamManager<float>::Instance().submitParallel(MatrixSubKernel, gridDim, blockDim, 0, c, a, b, offset_c,
                                                        offset_a, offset_b, n);
    }

    __global__ void GeneralMatrixMulKernel(float* C, const float* A, const float* B,
                                           const unsigned long long M,
                                           const unsigned long long N,
                                           const unsigned long long K,
                                           const size_t offset_c,
                                           const size_t offset_a,
                                           const size_t offset_b) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        const unsigned long long row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned long long col = blockIdx.x * blockDim.x + threadIdx.x;

        float sum = 0.0f;

        for (int t = 0; t < K; t += TILE_SIZE) {
            if (row < M && t + threadIdx.x < K)
                As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x + offset_a];
            else
                As[threadIdx.y][threadIdx.x] = 0.0f;

            if (col < N && t + threadIdx.y < K)
                Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col + offset_b];
            else
                Bs[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();

            for (int i = 0; i < TILE_SIZE; i++)
                sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

            __syncthreads();
        }

        if (row < M && col < N)
            C[row * N + col + offset_c] = sum;
    }

    void GeneralMatrixMul(const dim3 gridDim, const dim3 blockDim, float* A, float* B, float* C,
                          const unsigned long long M,
                          const unsigned long long N,
                          const unsigned long long K,
                          const size_t offset_c,
                          const size_t offset_a,
                          const size_t offset_b) {
        StreamManager<float>::Instance().submit(GeneralMatrixMulKernel, gridDim, blockDim, 0, C, A, B, M, N, K,
                                                offset_c, offset_a, offset_b);
    }

    void GeneralMatrixMul(const dim3 gridDim, const dim3 blockDim, float* A, float* B, float* C,
                          const unsigned long long M,
                          const unsigned long long N,
                          const unsigned long long K,
                          const std::vector<size_t>& offset_c,
                          const std::vector<size_t>& offset_a,
                          const std::vector<size_t>& offset_b) {
        StreamManager<float>::Instance().submitParallel(GeneralMatrixMulKernel, gridDim, blockDim, 0, C, A, B,
                                                        offset_c, offset_a, offset_b, M, N, K);
    }

    __global__ void TransposeKernel(float* d_B, const float* d_A,
                                    const unsigned int rows,
                                    const unsigned int cols,
                                    const size_t offset) {
        __shared__ float tile[TILE_SIZE][TILE_SIZE];

        unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;

        if (row < rows && col < cols)
            tile[threadIdx.y][threadIdx.x] = d_A[row * cols + col + offset];
        else
            tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        row = blockIdx.x * TILE_SIZE + threadIdx.y;
        col = blockIdx.y * TILE_SIZE + threadIdx.x;
        if (row < cols && col < rows)
            d_B[row * rows + col + offset] = tile[threadIdx.x][threadIdx.y];
    }

    void Transpose(const dim3 gridDim, const dim3 blockDim, float* d_A, float* d_B,
                   const unsigned int rows,
                   const unsigned int cols,
                   const size_t offset) {
        StreamManager<float>::Instance().submit(TransposeKernel, gridDim, blockDim, 0, d_B, d_A, rows, cols, offset);
    }

    void Transpose(const dim3 gridDim, const dim3 blockDim, float* d_A, float* d_B,
                   const unsigned int rows,
                   const unsigned int cols,
                   const std::vector<size_t>& offset) {
        StreamManager<float>::Instance().submitParallel(TransposeKernel, gridDim, blockDim, 0, d_B, d_A, offset, rows,
                                                        cols);
    }

    __global__ void ScalarMulKernel(float* out, const float* in, const float num,
                                    const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] * num;
        }
    }

    void ScalarMul(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const float num,
                   const unsigned long long n) {
        StreamManager<float>::Instance().submit(ScalarMulKernel, gridDim, blockDim, 0, out, in, num, n);
    }

    __global__ void ScalarDivKernel(float* out, const float* in, const float num,
                                    const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] / num;
        }
    }

    void ScalarDiv(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const float num,
                   const unsigned long long n) {
        StreamManager<float>::Instance().submit(ScalarDivKernel, gridDim, blockDim, 0, out, in, num, n);
    }

    __global__ void ScalarAddKernel(float* out, const float* in, const float num,
                                    const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] + num;
        }
    }

    void ScalarAdd(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const float num,
                   const unsigned long long n) {
        StreamManager<float>::Instance().submit(ScalarAddKernel, gridDim, blockDim, 0, out, in, num, n);
    }

    __global__ void NegationKernel(float* out, const float* in,
                                   const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = -in[idx];
        }
    }

    void Negation(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const unsigned long long n) {
        StreamManager<float>::Instance().submit(NegationKernel, gridDim, blockDim, 0, out, in, n);
    }

    __global__ void
    RecipKernel(float* out, const float* in, const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            if (in[idx] == 0) {
                out[idx] = 0.0f;
            }
            else {
                out[idx] = 1.0f / in[idx];
            }
        }
    }

    void Recip(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const unsigned long long n) {
        StreamManager<float>::Instance().submit(RecipKernel, gridDim, blockDim, 0, out, in, n);
    }

    __global__ void RectifiedLinearUnitKernel(float* out, const float* in, const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] > 0 ? in[idx] : 0;
        }
    }

    void RectifiedLinearUnit(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                             const unsigned long long n) {
        StreamManager<float>::Instance().submit(RectifiedLinearUnitKernel, gridDim, blockDim, 0, out, in, n);
    }

    __global__ void ReLUBackwardKernel(float* A_grad, const float* A,
                                       const float* B_grad, const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = A[idx] > 0 ? B_grad[idx] : 0;
        }
    }

    void ReLUBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, float* A, float* B_grad,
                      const unsigned long long n) {
        StreamManager<float>::Instance().submit(ReLUBackwardKernel, gridDim, blockDim, 0, A_grad, A, B_grad, n);
    }

    __global__ void SigmoidKernel(float* out, const float* in,
                                  const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = 1.0f / (1.0f + __expf(-in[idx]));
        }
    }

    void Sigmoid(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                 const unsigned long long n) {
        StreamManager<float>::Instance().submit(SigmoidKernel, gridDim, blockDim, 0, out, in, n);
    }

    __global__ void SigmoidBackwardKernel(float* A_grad, const float* B,
                                          const float* B_grad,
                                          unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = B[idx] * (1.0f - B[idx]) * B_grad[idx];
        }
    }

    void SigmoidBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, float* B, float* B_grad,
                         const unsigned long long n) {
        StreamManager<float>::Instance().submit(SigmoidBackwardKernel, gridDim, blockDim, 0, A_grad, B, B_grad, n);
    }

    __global__ void TanhKernel(float* out, const float* in, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (__expf(in[idx]) - __expf(-in[idx])) / (__expf(in[idx]) + __expf(-in[idx]));
        }
    }

    void Tanh(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
              const unsigned long long n) {
        StreamManager<float>::Instance().submit(TanhKernel, gridDim, blockDim, 0, out, in, n);
    }

    __global__ void TanhBackwardKernel(float* A_grad, const float* B,
                                       const float* B_grad, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = (1.0f - B[idx] * B[idx]) * B_grad[idx];
        }
    }

    void TanhBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, float* B, float* B_grad,
                      const unsigned long long n) {
        StreamManager<float>::Instance().submit(TanhBackwardKernel, gridDim, blockDim, 0, A_grad, B, B_grad, n);
    }

    __global__ void LeakyReLUKernel(float* out, const float* in,
                                    const unsigned long long n, const float alpha) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] > 0 ? in[idx] : alpha * in[idx];
        }
    }

    void LeakyReLU(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                   const unsigned long long n, const float alpha) {
        StreamManager<float>::Instance().submit(LeakyReLUKernel, gridDim, blockDim, 0, out, in, n, alpha);
    }

    __global__ void LeakyReLUBackwardKernel(float* A_grad, const float* A,
                                            const float* B_grad,
                                            const unsigned long long n,
                                            const float alpha) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = A[idx] > 0 ? B_grad[idx] : alpha * B_grad[idx];
        }
    }

    void LeakyReLUBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, float* A, float* B_grad,
                           const unsigned long long n, const float alpha) {
        StreamManager<float>::Instance().submit(LeakyReLUBackwardKernel, gridDim, blockDim, 0, A_grad, A, B_grad, n,
                                                alpha);
    }

    __global__ void
    SwishKernel(float* out, const float* in, const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] / (1.0f + __expf(-in[idx]));
        }
    }

    void Swish(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
               const unsigned long long n) {
        StreamManager<float>::Instance().submit(SwishKernel, gridDim, blockDim, 0, out, in, n);
    }

    __global__ void SwishBackwardKernel(float* A_grad, const float* A,
                                        const float* B, const float* B_grad,
                                        const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = 1.0f / (1.0f + __expf(-A[idx])) + B[idx] * (1.0f - B[idx]) *
                B_grad[idx];
        }
    }

    void SwishBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, float* A, float* B,
                       float* B_grad, const unsigned long long n) {
        StreamManager<float>::Instance().submit(SwishBackwardKernel, gridDim, blockDim, 0, A_grad, A, B, B_grad, n);
    }

    __global__ void ExponentialLinearUnitKernel(float* out, const float* in, const unsigned long long n,
                                                const float alpha) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] > 0 ? in[idx] : alpha * (__expf(in[idx]) - 1);
        }
    }

    void ExponentialLinearUnit(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                               const unsigned long long n, const float alpha) {
        StreamManager<float>::Instance().submit(ExponentialLinearUnitKernel, gridDim, blockDim, 0, out, in, n, alpha);
    }

    __global__ void ELUBackwardKernel(float* A_grad, const float* A,
                                      const float* B_grad, const unsigned long long n,
                                      const float alpha) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = A[idx] > 0
                              ? B_grad[idx]
                              : alpha * __expf(A[idx]) * B_grad[idx];
        }
    }

    void ELUBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, float* A, float* B_grad,
                     const unsigned long long n, const float alpha) {
        StreamManager<float>::Instance().submit(ELUBackwardKernel, gridDim, blockDim, 0, A_grad, A, B_grad, n, alpha);
    }

    __global__ void HardSigmoidKernel(float* out, const float* in,
                                      const unsigned long long n, const float alpha,
                                      const float beta) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] * alpha + beta;
            out[idx] = out[idx] > 1.0f ? 1.0f : (out[idx] < 0.0f ? 0.0f : out[idx]);
        }
    }

    void HardSigmoid(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                     const unsigned long long n, const float alpha, const float beta) {
        StreamManager<float>::Instance().submit(HardSigmoidKernel, gridDim, blockDim, 0, out, in, n, alpha, beta);
    }

    __global__ void HardSigmoidBackwardKernel(float* A_grad, const float* A,
                                              const float* B_grad,
                                              const unsigned long long n,
                                              const float alpha, const float beta) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x = A[idx] * alpha + beta;
            if (x > 0.0f && x < 1.0f) {
                A_grad[idx] = B_grad[idx] * alpha;
            }
            else {
                A_grad[idx] = 0.0f;
            }
        }
    }

    void HardSigmoidBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, float* A,
                             float* B_grad,
                             const unsigned long long n, const float alpha, const float beta) {
        StreamManager<float>::Instance().submit(HardSigmoidBackwardKernel, gridDim, blockDim, 0, A_grad, A, B_grad, n,
                                                alpha, beta);
    }

    __inline__ __device__ float LiteHardSigmoid(float x, float alpha, float beta) {
        float a = x * alpha + beta;
        return a > 1.0f ? 1.0f : (a < 0.0f ? 0.0f : a);
    }

    __global__ void HardSwishKernel(float* out, const float* in,
                                    const unsigned long long n, const float alpha,
                                    const float beta) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] * LiteHardSigmoid(in[idx], alpha, beta);
        }
    }

    void HardSwish(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                   const unsigned long long n, const float alpha, const float beta) {
        StreamManager<float>::Instance().submit(HardSwishKernel, gridDim, blockDim, 0, out, in, n, alpha, beta);
    }

    __global__ void HardSwishBackwardKernel(float* A_grad, const float* A,
                                            const float* B_grad,
                                            const unsigned long long n,
                                            const float alpha, const float beta) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = LiteHardSigmoid(A[idx], alpha, beta) + B_grad[idx] * A[idx] *
                alpha * (1 - LiteHardSigmoid(
                    A[idx], alpha, beta));
        }
    }

    void HardSwishBackward(const dim3 gridDim, const dim3 blockDim, float* A_grad, float* A, float* B_grad,
                           const unsigned long long n, const float alpha, const float beta) {
        StreamManager<float>::Instance().submit(HardSwishBackwardKernel, gridDim, blockDim, 0, A_grad, A, B_grad, n,
                                                alpha, beta);
    }

    __inline__ __device__ float warpReduce(float localSum) {
        localSum += __shfl_xor_sync(FULL_MASK, localSum, 16);
        localSum += __shfl_xor_sync(FULL_MASK, localSum, 8);
        localSum += __shfl_xor_sync(FULL_MASK, localSum, 4);
        localSum += __shfl_xor_sync(FULL_MASK, localSum, 2);
        localSum += __shfl_xor_sync(FULL_MASK, localSum, 1);
        return localSum;
    }

    __global__ void SummationExpKernel(float* out, const float* g_data,
                                       const unsigned long long n, const size_t offset) {
        extern __shared__ float sdata[];
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long tid = threadIdx.x;
        const unsigned long long warpIdx = tid / WARP_SIZE;
        const unsigned long long laneIdx = tid % WARP_SIZE;
        float localSum = 0.0f;
        if (idx < n) {
            localSum = __expf(g_data[idx + offset]);
        }
        else {
            localSum = 0.0f;
        }
        __syncthreads();
        // Warp Reduce
        localSum = warpReduce(localSum);
        if (laneIdx == 0) {
            sdata[warpIdx] = localSum;
        }
        __syncthreads();
        localSum = (tid < blockDim.x / WARP_SIZE) ? sdata[laneIdx] : 0.0f;
        // Block Reduce
        if (warpIdx == 0) {
            localSum = warpReduce(localSum);
        }
        __syncthreads();

        if (tid == 0) {
            out[blockIdx.x] = localSum;
        }
    }

    void SummationExp(const dim3 gridDim, const dim3 blockDim, const size_t sharedMemSize, float* out,
                      float* g_data,
                      const unsigned long long n, const size_t offset) {
        StreamManager<float>::Instance().submit(SummationExpKernel, gridDim, blockDim, sharedMemSize, out, g_data, n,
                                                offset);
    }

    __global__ void SoftmaxKernel(float* out, const float* in,
                                  const float exp_sum_of_input, const unsigned long long n, const size_t offset) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx + offset] = __expf(in[idx + offset]) / exp_sum_of_input;
        }
    }

    void Softmax(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const float exp_sum_of_input,
                 const unsigned long long n, const size_t offset) {
        StreamManager<float>::Instance().submit(SoftmaxKernel, gridDim, blockDim, 0, out, in, exp_sum_of_input, n,
                                                offset);
    }

    void Softmax(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                 const std::vector<float>& exp_sum_of_input,
                 const unsigned long long n, const std::vector<size_t>& offset) {
        if (exp_sum_of_input.size() != offset.size()) {
            throw std::runtime_error("exp_sum_of_input and offset must have the same size");
        }
        std::vector<cudaStream_t> streams;
        for (auto i = 0; i < exp_sum_of_input.size(); i++) {
            auto stream = StreamManager<float>::Instance().getStream();
            StreamManager<float>::Instance().streamWait(out, stream);
            StreamManager<float>::Instance().streamWait(in, stream);
            streams.push_back(stream);
            SoftmaxKernel<<<gridDim, blockDim, 0, stream>>>(out, in, exp_sum_of_input[i], n, offset[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(out, stream);
        }
    }

    __global__ void SoftmaxJacobianKernel(float* out, const float* in,
                                          const unsigned long long n, const size_t offset_o = 0,
                                          const size_t offset_i = 0) {
        unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long long idy = blockIdx.y * blockDim.y + threadIdx.y;
        if (idx >= n || idy >= n) {
            return;
        }
        unsigned long long id = idx * n + idy;
        if (idy == idx) {
            out[id + offset_o] = in[idx + offset_i] * (1 - in[idx + offset_i]);
        }
        else {
            out[id + offset_o] = -in[idx + offset_i] * in[idy + offset_i];
        }
    }

    void SoftmaxJacobian(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                         const unsigned long long n) {
        StreamManager<float>::Instance().submit(SoftmaxJacobianKernel, gridDim, blockDim, 0, out, in, n, 0, 0);
    }

    void SoftmaxJacobian(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
                         const unsigned long long n, const std::vector<size_t>& offset_o,
                         const std::vector<size_t>& offset_i) {
        StreamManager<float>::Instance().submitParallel(SoftmaxJacobianKernel, gridDim, blockDim, 0, out, in, offset_o,
                                                        offset_i, n);
    }

    __global__ void MeanSquaredErrorKernel(float* out, const float* predict, const float* real,
                                           const unsigned long long n) {
        extern __shared__ float smem[];
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long tid = threadIdx.x;
        const unsigned long long warpIdx = tid / WARP_SIZE;
        const unsigned long long laneIdx = tid % WARP_SIZE;
        float localSum = 0.0f;
        if (idx < n) {
            localSum = (predict[idx] - real[idx]) * (predict[idx] - real[idx]) / (float)n;
        }
        else {
            localSum = 0.0f;
        }
        localSum = warpReduce(localSum);
        __syncthreads();

        if (laneIdx == 0) {
            smem[warpIdx] = localSum;
        }
        __syncthreads();

        localSum = (tid < blockDim.x / WARP_SIZE) ? smem[laneIdx] : 0.0f;

        __syncthreads();

        if (warpIdx == 0) {
            localSum = warpReduce(localSum);
        }
        __syncthreads();

        if (tid == 0) {
            out[blockIdx.x] = localSum;
        }
    }

    void MeanSquaredError(const dim3 gridDim, const dim3 blockDim, const size_t sharedMemSize, float* out,
                          float* predict, float* real, const unsigned long long n) {
        StreamManager<float>::Instance().submit(MeanSquaredErrorKernel, gridDim, blockDim, sharedMemSize, out, predict,
                                                real, n);
    }

    __global__ void MSEBackwardKernel(float* out, const float* predict,
                                      const float* real, const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = 2 * (predict[idx] - real[idx]) / (float)n;
        }
    }

    void MSEBackward(const dim3 gridDim, const dim3 blockDim, float* out, float* predict,
                     float* real, const unsigned long long n) {
        StreamManager<float>::Instance().submit(MSEBackwardKernel, gridDim, blockDim, 0, out, predict, real, n);
    }

    __global__ void StochasticGradientDescentKernel(float* data, const float* grad, const float lr,
                                                    const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] -= lr * grad[idx];
        }
    }

    void StochasticGradientDescent(const dim3 gridDim, const dim3 blockDim, float* data, float* grad,
                                   const float lr, const unsigned long long n) {
        StreamManager<float>::Instance().submit(StochasticGradientDescentKernel, gridDim, blockDim, 0, data, grad, lr,
                                                n);
    }

    __global__ void BinaryCrossEntropyKernel(float* out, const float* predict, const float* real,
                                             const unsigned long long n) {
        extern __shared__ float smem[];
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long tid = threadIdx.x;
        const unsigned long long warpIdx = tid / WARP_SIZE;
        const unsigned long long laneIdx = tid % WARP_SIZE;
        float localSum = 0.0f;
        if (idx < n) {
            localSum = (-real[idx] * __logf(predict[idx]) - (1 - real[idx]) *
                __logf(1 - predict[idx])) / (float)n;
        }
        else {
            localSum = 0;
        }
        __syncthreads();

        localSum = warpReduce(localSum);
        __syncthreads();

        if (laneIdx == 0) {
            smem[warpIdx] = localSum;
        }
        __syncthreads();

        localSum = (tid < blockDim.x / WARP_SIZE) ? smem[laneIdx] : 0.0f;
        __syncthreads();

        if (warpIdx == 0) {
            localSum = warpReduce(localSum);
        }
        __syncthreads();

        if (tid == 0) {
            out[blockIdx.x] = localSum;
        }
    }

    void BinaryCrossEntropy(const dim3 gridDim, const dim3 blockDim, const size_t sharedMemSize, float* out,
                            float* predict, float* real, const unsigned long long n) {
        StreamManager<float>::Instance().submit(BinaryCrossEntropyKernel, gridDim, blockDim, sharedMemSize, out,
                                                predict, real, n);
    }

    __global__ void BCEBackwardKernel(float* out, const float* predict,
                                      const float* real, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = ((predict[idx] - real[idx]) / (
                predict[idx] * (1 - predict[idx]))) / (float)n;
        }
    }

    void BCEBackward(const dim3 gridDim, const dim3 blockDim, float* out, float* predict,
                     float* real, const unsigned long long n) {
        StreamManager<float>::Instance().submit(BCEBackwardKernel, gridDim, blockDim, 0, out, predict, real, n);
    }

    __global__ void MomentumKernel(float* output, const float* grad,
                                   const float* velocity, const float beta,
                                   const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = velocity[idx] * beta + grad[idx] * (1 - beta);
        }
    }

    void Momentum(const dim3 gridDim, const dim3 blockDim, float* output, float* grad, float* velocity,
                  const float beta, const unsigned long long n) {
        StreamManager<float>::Instance().submit(MomentumKernel, gridDim, blockDim, 0, output, grad, velocity, beta, n);
    }

    __global__ void AdaGradKernel(float* data, float* G, const float* grad, const float lr, const float eps,
                                  const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float temp = G[idx] + grad[idx] * grad[idx];
        data[idx] -= lr * grad[idx] / (sqrtf(temp) + eps);
        G[idx] = temp;
    }

    void AdaGrad(const dim3 gridDim, const dim3 blockDim, float* data, float* G, float* grad, const float lr,
                 const float eps, const unsigned long long n) {
        StreamManager<float>::Instance().submitDualOut(AdaGradKernel, gridDim, blockDim, 0, data, G, grad, lr, eps, n);
    }

    __global__ void RMSpropKernel(float* data, float* v, const float* grad, const float lr, const float beta,
                                  const float eps, const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float temp = v[idx] * beta + grad[idx] * grad[idx] * (1 - beta);
        data[idx] -= lr * grad[idx] / (sqrtf(temp) + eps);
        v[idx] = temp;
    }

    void RMSprop(const dim3 gridDim, const dim3 blockDim, float* data, float* v, float* grad, const float lr,
                 const float beta, const float eps, const unsigned long long n) {
        StreamManager<float>::Instance().submitDualOut(RMSpropKernel, gridDim, blockDim, 0, data, v, grad, lr, beta,
                                                       eps, n);
    }

    __global__ void AdamKernel(float* data, float* m, float* v, const float* grad, const float lr, const float beta1,
                               const float beta2, const float eps, const int t, const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float m_temp = m[idx] * beta1 + grad[idx] * (1 - beta1);
        const float v_temp = v[idx] * beta2 + grad[idx] * grad[idx] * (1 - beta2);
        const float m_modified = m_temp / (1 - __powf(beta1, (float)t));
        const float v_modified = v_temp / (1 - __powf(beta2, (float)t));
        data[idx] -= lr * m_modified / (sqrtf(v_modified) + eps);
        m[idx] = m_temp;
        v[idx] = v_temp;
    }

    void Adam(const dim3 gridDim, const dim3 blockDim, float* data, float* m, float* v, float* grad,
              const float lr, const float beta1, const float beta2, const float eps, const int t,
              const unsigned long long n) {
        StreamManager<float>::Instance().submitTripleOut(AdamKernel, gridDim, blockDim, 0, data, m, v, grad, lr, beta1,
                                                         beta2, eps, t, n);
    }

    __global__ void NAdamKernel(float* data, float* m, float* m_modified, float* v, const float* grad, const float lr,
                                const float beta1, const float beta2, const float eps, const int t,
                                const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float m_temp = m[idx] * beta1 + grad[idx] * (1 - beta1);
        const float v_temp = v[idx] * beta2 + grad[idx] * grad[idx] * (1 - beta2);
        const float m_temp_modified = m_temp / (1 - __powf(beta1, (float)t));
        const float v_modified = v_temp / (1 - __powf(beta2, (float)t));
        const float m_modified_minus_1 = m_modified[idx] * beta1 + grad[idx] * (1 - beta1);
        data[idx] -= lr * m_modified_minus_1 / (sqrtf(v_modified) + eps);
        m[idx] = m_temp;
        m_modified[idx] = m_temp_modified;
        v[idx] = v_temp;
    }

    void NAdam(const dim3 gridDim, const dim3 blockDim, float* data, float* m, float* m_modified, float* v,
               float* grad, const float lr, const float beta1, const float beta2, const float eps, const int t,
               const unsigned long long n) {
        StreamManager<float>::Instance().submitQuadOut(NAdamKernel, gridDim, blockDim, 0, data, m, m_modified, v, grad,
                                                       lr, beta1, beta2, eps, t, n);
    }

    __global__ void AdaDeltaKernel(float* data, float* acc_delta, float* acc_grad, const float* grad,
                                   const float rho, const float eps,
                                   const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float delta_acc_grad_temp = acc_grad[idx] * rho + grad[idx] * grad[idx] * (1 - rho);
        const float delta_theta = -grad[idx] * sqrtf(acc_delta[idx] + eps) / sqrtf(delta_acc_grad_temp + eps);
        data[idx] += delta_theta;
        const float delta_acc_temp = acc_delta[idx] * rho + delta_theta * delta_theta * (1 - rho);
        acc_delta[idx] = delta_acc_temp;
        acc_grad[idx] = delta_acc_grad_temp;
    }

    void AdaDelta(const dim3 gridDim, const dim3 blockDim, float* data, float* acc_delta, float* acc_grad,
                  float* grad, const float rho, const float eps, const unsigned long long n) {
        StreamManager<float>::Instance().submitTripleOut(AdaDeltaKernel, gridDim, blockDim, 0, data, acc_delta,
                                                         acc_grad, grad, rho, eps, n);
    }

    __global__ void GeneralMatrixMulTensorKernel(float* C, const half* A, const half* B, const unsigned long long m,
                                                 const unsigned long long n, const unsigned long long k,
                                                 size_t offset_c = 0,
                                                 size_t offset_a = 0,
                                                 size_t offset_b = 0) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long warpIdx = idx / warpSize;
        const unsigned long long blockM = m / MMA;
        const unsigned long long blockN = n / MMA;
        const unsigned long long rowA = warpIdx / blockN;
        const unsigned long long colB = warpIdx % blockN;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MMA, MMA, MMA, half, nvcuda::wmma::row_major> A_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MMA, MMA, MMA, half, nvcuda::wmma::row_major> B_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMA, MMA, MMA, float> C_frag;
        fill_fragment(C_frag, 0.0f);
        if (rowA < blockM && colB < blockN) {
            for (int i = 0; i < k / MMA; i++) {
                load_matrix_sync(A_frag, A + offset_a + rowA * k * MMA + i * MMA, k);
                load_matrix_sync(B_frag, B + offset_b + colB * MMA + i * n * MMA, n);
                mma_sync(C_frag, A_frag, B_frag, C_frag);
            }
            store_matrix_sync(C + offset_c + rowA * n * MMA + colB * MMA, C_frag, n, nvcuda::wmma::mem_row_major);
        }
    }

    __global__ void Padding(half* out, const float* in, const unsigned long long M, const unsigned long long N,
                            const unsigned long long m, const unsigned long long n, size_t offset_o = 0,
                            size_t offset_i = 0) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long row = idx / n;
        const unsigned long long col = idx % n;
        if (row < m && col < n) {
            if (row < M && col < N) {
                out[row * n + col + offset_o] = __float2half(in[row * N + col + offset_i]);
            }
            else {
                out[row * n + col + offset_o] = __float2half(0.0f);
            }
        }
    }

    __global__ void Cutting(float* out, const float* in, const unsigned long long M, const unsigned long long N,
                            const unsigned long long m, const unsigned long long n, size_t offset_o = 0,
                            size_t offset_i = 0) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long row = idx / n;
        const unsigned long long col = idx % n;
        if (row < M && col < N) {
            out[row * N + col + offset_o] = in[row * n + col + offset_i];
        }
    }

    __global__ void CuttingAndCompress(float* out, const float* in, const unsigned long long M,
                                       const unsigned long long N,
                                       const unsigned long long m, const unsigned long long n, size_t offset_o = 0,
                                       size_t offset_i = 0) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long row = idx / n;
        const unsigned long long col = idx % n;
        if (row < M && col < N) {
            atomicAdd(out + (row * N + col + offset_o), in[row * n + col + offset_i]);
        }
    }

    void TensorCoreGEMM(float* A, float* B, float* C, const unsigned long long M,
                        const unsigned long long N, const unsigned long long K) {
        const unsigned long long m = CEIL(M);
        const unsigned long long k = CEIL(K);
        const unsigned long long n = CEIL(N);
        half* padded_A;
        half* padded_B;
        float* padded_C;
        StreamManager<float>::Instance().malloc(&padded_A, m * k * sizeof(half));
        StreamManager<float>::Instance().malloc(&padded_B, k * n * sizeof(half));
        StreamManager<float>::Instance().malloc(&padded_C, m * n * sizeof(float));
        StreamManager<float>::Instance().memset(padded_C, 0, m * n * sizeof(float));
        StreamManager<float>::Instance().submit(Padding, dim3((m * k + 256 - 1) / 256), dim3(256), 0, padded_A, A, M, K,
                                                m, k, 0, 0);
        StreamManager<float>::Instance().submit(Padding, dim3((k * n + 256 - 1) / 256), dim3(256), 0, padded_B, B, K, N,
                                                k, n, 0, 0);
        const unsigned long long tiles = (m * n) >> 8;
        dim3 block(256);
        const unsigned int warpPerBlock = block.x / 32;
        dim3 grid((tiles + warpPerBlock - 1) / warpPerBlock);
        StreamManager<float>::Instance().submit(GeneralMatrixMulTensorKernel, grid, block, 0, padded_C, padded_A,
                                                padded_B, m, n, k, 0, 0, 0);
        StreamManager<float>::Instance().submit(Cutting, dim3((m * n + 256 - 1) / 256), dim3(256), 0, C, padded_C, M, N,
                                                m, n, 0, 0);
        StreamManager<float>::Instance().freeAsync(padded_A);
        StreamManager<float>::Instance().freeAsync(padded_B);
        StreamManager<float>::Instance().freeAsync(padded_C);
    }

    void TensorCoreGEMMParallel(float* A, float* B, float* C,
                                const data::Dimension& A_shape,
                                const data::Dimension& B_shape,
                                const data::Dimension& C_shape) {
        const size_t M = A_shape.H();
        const size_t N = B_shape.W();
        const size_t K = A_shape.W();
        const size_t m = CEIL(M);
        const size_t n = CEIL(N);
        const size_t k = CEIL(K);
        const auto padded_A_shape = data::Dimension(A_shape.N(), A_shape.C(), m, k);
        const auto padded_B_shape = data::Dimension(B_shape.N(), B_shape.C(), k, n);
        const auto padded_C_shape = data::Dimension(C_shape.N(), C_shape.C(), m, n);
        half* padded_A;
        half* padded_B;
        float* padded_C;
        StreamManager<float>::Instance().malloc(&padded_A, padded_A_shape.size() * sizeof(half));
        StreamManager<float>::Instance().malloc(&padded_B, padded_B_shape.size() * sizeof(half));
        StreamManager<float>::Instance().malloc(&padded_C, padded_C_shape.size() * sizeof(float));
        StreamManager<float>::Instance().memset(padded_C, 0, padded_C_shape.size() * sizeof(float));
        std::vector<size_t> offset1;
        std::vector<size_t> offset2;
        for (auto i = 0; i < A_shape.N(); i++) {
            for (auto j = 0; j < A_shape.C(); j++) {
                offset1.push_back(i * A_shape.getStride(0) + j * A_shape.getStride(1));
                offset2.push_back(i * padded_A_shape.getStride(0) + j * padded_A_shape.getStride(1));
            }
        }
        std::vector<cudaStream_t> streams;
        for (auto i = 0; i < offset1.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(A, stream);
            StreamManager<float>::Instance().streamWait(padded_A, stream);
            Padding<<<dim3((m * k + 256 - 1) / 256), dim3(256), 0, stream>>>(
                padded_A, A, M, K, m, k, offset2[i], offset1[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(padded_A, stream);
        }
        offset1.clear();
        offset2.clear();
        streams.clear();
        for (auto i = 0; i < B_shape.N(); i++) {
            for (auto j = 0; j < B_shape.C(); j++) {
                offset1.push_back(i * B_shape.getStride(0) + j * B_shape.getStride(1));
                offset2.push_back(i * padded_B_shape.getStride(0) + j * padded_B_shape.getStride(1));
            }
        }
        for (auto i = 0; i < offset1.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(B, stream);
            StreamManager<float>::Instance().streamWait(padded_B, stream);
            Padding<<<dim3((k * n + 256 - 1) / 256), dim3(256), 0, stream>>>(
                padded_B, B, K, N, k, n, offset2[i], offset1[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(padded_B, stream);
        }
        offset1.clear();
        offset2.clear();
        streams.clear();
        std::vector<size_t> offset3;
        for (auto i = 0; i < padded_C_shape[0]; i++) {
            for (auto j = 0; j < padded_C_shape[1]; j++) {
                offset3.push_back(i * padded_C_shape.getStride(0) + j * padded_C_shape.getStride(1));
                offset1.push_back(i * (padded_A_shape[0] > 1 ? padded_A_shape.getStride(0) : 0) +
                    j * (padded_A_shape[1] > 1 ? padded_A_shape.getStride(1) : 0));
                offset2.push_back(i * (padded_B_shape[0] > 1 ? padded_B_shape.getStride(0) : 0) +
                    j * (padded_B_shape[1] > 1 ? padded_B_shape.getStride(1) : 0));
            }
        }
        const unsigned long long tiles = (m * n) >> 8;
        dim3 block(256);
        const unsigned int warpPerBlock = block.x / 32;
        dim3 grid((tiles + warpPerBlock - 1) / warpPerBlock);
        for (auto i = 0; i < offset3.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(padded_C, stream);
            StreamManager<float>::Instance().streamWait(padded_A, stream);
            StreamManager<float>::Instance().streamWait(padded_B, stream);
            GeneralMatrixMulTensorKernel<<<grid, block, 0, stream>>>(padded_C, padded_A, padded_B, m, n, k, offset3[i],
                                                                     offset1[i], offset2[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(padded_C, stream);
        }
        offset1.clear();
        offset2.clear();
        streams.clear();
        for (auto i = 0; i < C_shape.N(); i++) {
            for (auto j = 0; j < C_shape.C(); j++) {
                offset1.push_back(i * C_shape.getStride(0) + j * C_shape.getStride(1));
                offset2.push_back(i * padded_C_shape.getStride(0) + j * padded_C_shape.getStride(1));
            }
        }
        for (auto i = 0; i < offset1.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(C, stream);
            StreamManager<float>::Instance().streamWait(padded_C, stream);
            Cutting<<<dim3((n * m + 256 - 1) / 256), dim3(256), 0, stream>>>(
                C, padded_C, M, N, m, n, offset1[i], offset2[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(C, stream);
        }
        StreamManager<float>::Instance().freeAsync(padded_A);
        StreamManager<float>::Instance().freeAsync(padded_B);
        StreamManager<float>::Instance().freeAsync(padded_C);
    }

    void GEMMBackwardParallel(float* A, float* B, float* C,
                              const data::Dimension& A_shape,
                              const data::Dimension& B_shape,
                              const data::Dimension& C_shape) {
        const size_t M = A_shape.H();
        const size_t N = B_shape.W();
        const size_t K = A_shape.W();
        const size_t m = CEIL(M);
        const size_t n = CEIL(N);
        const size_t k = CEIL(K);
        const auto padded_A_shape = data::Dimension(A_shape.N(), A_shape.C(), m, k);
        const auto padded_B_shape = data::Dimension(B_shape.N(), B_shape.C(), k, n);
        const auto padded_C_shape = data::Dimension(std::max(A_shape.N(), B_shape.N()),
                                                    std::max(A_shape.C(), B_shape.C()), m, n);
        half* padded_A;
        half* padded_B;
        float* padded_C;
        StreamManager<float>::Instance().malloc(&padded_A, padded_A_shape.size() * sizeof(half));
        StreamManager<float>::Instance().malloc(&padded_B, padded_B_shape.size() * sizeof(half));
        StreamManager<float>::Instance().malloc(&padded_C, padded_C_shape.size() * sizeof(float));
        StreamManager<float>::Instance().memset(padded_C, 0, padded_C_shape.size() * sizeof(float));
        std::vector<size_t> offset1;
        std::vector<size_t> offset2;
        for (auto i = 0; i < A_shape.N(); i++) {
            for (auto j = 0; j < A_shape.C(); j++) {
                offset1.push_back(i * A_shape.getStride(0) + j * A_shape.getStride(1));
                offset2.push_back(i * padded_A_shape.getStride(0) + j * padded_A_shape.getStride(1));
            }
        }
        std::vector<cudaStream_t> streams;
        for (auto i = 0; i < offset1.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(A, stream);
            StreamManager<float>::Instance().streamWait(padded_A, stream);
            Padding<<<dim3((m * k + 256 - 1) / 256), dim3(256), 0, stream>>>(
                padded_A, A, M, K, m, k, offset2[i], offset1[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(padded_A, stream);
        }
        offset1.clear();
        offset2.clear();
        streams.clear();
        for (auto i = 0; i < B_shape.N(); i++) {
            for (auto j = 0; j < B_shape.C(); j++) {
                offset1.push_back(i * B_shape.getStride(0) + j * B_shape.getStride(1));
                offset2.push_back(i * padded_B_shape.getStride(0) + j * padded_B_shape.getStride(1));
            }
        }
        for (auto i = 0; i < offset1.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(B, stream);
            StreamManager<float>::Instance().streamWait(padded_B, stream);
            Padding<<<dim3((k * n + 256 - 1) / 256), dim3(256), 0, stream>>>(
                padded_B, B, K, N, k, n, offset2[i], offset1[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(padded_B, stream);
        }
        offset1.clear();
        offset2.clear();
        streams.clear();
        std::vector<size_t> offset3;
        for (auto i = 0; i < padded_C_shape[0]; i++) {
            for (auto j = 0; j < padded_C_shape[1]; j++) {
                offset3.push_back(i * padded_C_shape.getStride(0) + j * padded_C_shape.getStride(1));
                offset1.push_back(i * (padded_A_shape[0] > 1 ? padded_A_shape.getStride(0) : 0) +
                    j * (padded_A_shape[1] > 1 ? padded_A_shape.getStride(1) : 0));
                offset2.push_back(i * (padded_B_shape[0] > 1 ? padded_B_shape.getStride(0) : 0) +
                    j * (padded_B_shape[1] > 1 ? padded_B_shape.getStride(1) : 0));
            }
        }
        const unsigned long long tiles = (m * n) >> 8;
        dim3 block(256);
        const unsigned int warpPerBlock = block.x / 32;
        dim3 grid((tiles + warpPerBlock - 1) / warpPerBlock);
        for (auto i = 0; i < offset3.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(padded_C, stream);
            StreamManager<float>::Instance().streamWait(padded_A, stream);
            StreamManager<float>::Instance().streamWait(padded_B, stream);
            GeneralMatrixMulTensorKernel<<<grid, block, 0, stream>>>(padded_C, padded_A, padded_B, m, n, k, offset3[i],
                                                                     offset1[i], offset2[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(padded_C, stream);
        }
        offset1.clear();
        offset2.clear();
        streams.clear();
        for (auto i = 0; i < padded_C_shape.N(); i++) {
            for (auto j = 0; j < padded_C_shape.C(); j++) {
                offset1.push_back(
                    i * (C_shape.N() > 1 ? C_shape.getStride(0) : 0) + j * (C_shape.C() > 1
                                                                                ? C_shape.getStride(1)
                                                                                : 0));
                offset2.push_back(i * padded_C_shape.getStride(0) + j * padded_C_shape.getStride(1));
            }
        }
        for (auto i = 0; i < offset1.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(C, stream);
            StreamManager<float>::Instance().streamWait(padded_C, stream);
            CuttingAndCompress<<<dim3((n * m + 256 - 1) / 256), dim3(256), 0, stream>>>(
                C, padded_C, M, N, m, n, offset1[i], offset2[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(C, stream);
        }
        StreamManager<float>::Instance().freeAsync(padded_A);
        StreamManager<float>::Instance().freeAsync(padded_B);
        StreamManager<float>::Instance().freeAsync(padded_C);
    }

    __global__ void FillKernel(float* data, const float value, const unsigned long long n, const size_t offset) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx + offset] = value;
        }
    }

    void Fill(const dim3 gridDim, const dim3 blockDim, float* data, const float value, const unsigned long long n,
              const size_t offset) {
        StreamManager<float>::Instance().submit(FillKernel, gridDim, blockDim, 0, data, value, n, offset);
    }

    __global__ void HadamardProductKernel(float* out, const float* in1, const float* in2, const unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in1[idx] * in2[idx];
        }
    }

    void HadamardProduct(const dim3 gridDim, const dim3 blockDim, float* out, float* in1, float* in2,
                         const unsigned long long n) {
        StreamManager<float>::Instance().submit(HadamardProductKernel, gridDim, blockDim, 0, out, in1, in2, n);
    }

    __global__ void ElementwiseDivideKernel(float* out, const float* in1, const float* in2,
                                            const unsigned long long n,
                                            const size_t offset_o,
                                            const size_t offset_1,
                                            const size_t offset_2) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx + offset_o] = in1[idx + offset_1] / in2[idx + offset_2];
        }
    }

    void ElementwiseDivide(const dim3 gridDim, const dim3 blockDim, float* out, float* in1, float* in2,
                           const unsigned long long n, const size_t offset_o,
                           const size_t offset_1,
                           const size_t offset_2) {
        StreamManager<float>::Instance().submit(ElementwiseDivideKernel, gridDim, blockDim, 0, out, in1, in2, n,
                                                offset_o, offset_1, offset_2);
    }

    void ElementwiseDivide(const dim3 gridDim, const dim3 blockDim, float* out, float* in1, float* in2,
                           const unsigned long long n, const std::vector<size_t>& offset_o,
                           const std::vector<size_t>& offset_1,
                           const std::vector<size_t>& offset_2) {
        StreamManager<float>::Instance().submitParallel(ElementwiseDivideKernel, gridDim, blockDim, 0, out, in1, in2,
                                                        offset_o, offset_1, offset_2, n);
    }

    __global__ void SummationKernel(float* out, const float* in, const unsigned long long n, const size_t offset) {
        extern __shared__ float sdata[];
        const unsigned long long tid = threadIdx.x;
        const unsigned long long idx = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned long long warpIdx = tid / WARP_SIZE;
        const unsigned long long laneIdx = tid % WARP_SIZE;
        float localSum = 0.0f;
        if (idx < n) {
            localSum = in[idx + offset];
        }
        __syncthreads();
        localSum = warpReduce(localSum);
        __syncthreads();
        if (laneIdx == 0) {
            sdata[warpIdx] = localSum;
        }
        __syncthreads();
        localSum = (tid < blockDim.x / WARP_SIZE) ? sdata[laneIdx] : 0.0f;
        __syncthreads();
        if (warpIdx == 0) {
            localSum = warpReduce(localSum);
        }
        __syncthreads();
        if (tid == 0) {
            out[blockIdx.x] = localSum;
        }
    }

    void Summation(const dim3 gridDim, const dim3 blockDim, const unsigned long long sharedMemSize, float* out,
                   float* in, const unsigned long long n, const size_t offset) {
        StreamManager<float>::Instance().submit(SummationKernel, gridDim, blockDim, sharedMemSize, out, in, n, offset);
    }

    __global__ void gradCopyKernel(float* out, const float* in, const size_t n, const size_t offset_o,
                                   const size_t offset_i) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx + offset_o] += in[idx + offset_i];
        }
    }

    void gradCopy(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const size_t n,
                  const std::vector<size_t>& offset_o, const std::vector<size_t>& offset_i) {
        if (offset_o.size() != offset_i.size()) {
            throw std::invalid_argument("offset size do not match");
        }
        std::vector<cudaStream_t> streams;
        for (auto i = 0; i < offset_i.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(out, stream);
            StreamManager<float>::Instance().streamWait(in, stream);
            gradCopyKernel<<<gridDim, blockDim, 0, stream>>>(out, in, n, offset_o[i], offset_i[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(out, stream);
        }
    }

    __global__ void NgradCopyKernel(float* out, const float* in, const size_t n, const size_t offset_o,
                                    const size_t offset_i) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx + offset_o] -= in[idx + offset_i];
        }
    }

    void NgradCopy(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const size_t n,
                   const std::vector<size_t>& offset_o, const std::vector<size_t>& offset_i) {
        if (offset_o.size() != offset_i.size()) {
            throw std::invalid_argument("offset size do not match");
        }
        std::vector<cudaStream_t> streams;
        for (auto i = 0; i < offset_i.size(); i++) {
            cudaStream_t stream = StreamManager<float>::Instance().getStream();
            streams.push_back(stream);
            StreamManager<float>::Instance().streamWait(out, stream);
            StreamManager<float>::Instance().streamWait(in, stream);
            NgradCopyKernel<<<gridDim, blockDim, 0, stream>>>(out, in, n, offset_o[i], offset_i[i]);
        }
        for (auto stream : streams) {
            StreamManager<float>::Instance().recordData(out, stream);
        }
    }

    __global__ void ExpandKernel(float* out, const float* in, const size_t n, const size_t total) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        out[idx] = in[idx % n];
    }

    void Expand(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const size_t n,
                const size_t total) {
        StreamManager<float>::Instance().submit(ExpandKernel, gridDim, blockDim, 0, out, in, n, total);
    }

    __global__ void CompressKernel(float* out, const float* in, const size_t n, const size_t total) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) {
            return;
        }
        atomicAdd(out + idx % n, in[idx]);
    }

    void Compress(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const size_t n,
                  const size_t total) {
        StreamManager<float>::Instance().submit(CompressKernel, gridDim, blockDim, 0, out, in, n, total);
    }

    __global__ void img2colKernel(float* out, const float* in, const size_t H_out, const size_t W_out, const size_t C,
                                  const size_t K_h, const size_t K_w, const size_t stride, const size_t pad,
                                  const size_t H_in, const size_t W_in, const size_t batch) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= H_out * W_out * C * K_h * K_w * batch) {
            return;
        }
        const size_t fixedIdx = idx % (H_out * W_out * C * K_h * K_w);
        const size_t currentBatch = idx / (H_out * W_out * C * K_h * K_w);
        const size_t k = fixedIdx / (C * K_h * K_w);
        const size_t m = fixedIdx % (C * K_h * K_w);
        const size_t c = m / (K_h * K_w);
        const long long h = (k / W_out) * stride - pad + (m % (K_h * K_w)) / K_w;
        const long long w = (k % W_out) * stride - pad + m % K_w;
        if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
            out[idx] = in[currentBatch * (C * H_in * W_in) + c * (H_in * W_in) + h * W_in + w];
        }
        else {
            out[idx] = 0;
        }
    }

    void img2col(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const size_t H_out,
                 const size_t W_out, const size_t C, const size_t K_h, const size_t K_w, const size_t stride,
                 const size_t pad, const size_t H_in, const size_t W_in, const size_t batch) {
        StreamManager<float>::Instance().submit(img2colKernel, gridDim, blockDim, 0, out, in, H_out, W_out, C,
                                                K_h, K_w, stride, pad, H_in, W_in, batch);
    }

    __global__ void img2colBackwardKernel(float* out, const float* in, const size_t H_out, const size_t W_out,
                                          const size_t C,
                                          const size_t K_h, const size_t K_w, const size_t stride, const size_t pad,
                                          const size_t H_in, const size_t W_in, const size_t batch) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= H_out * W_out * C * K_h * K_w * batch) {
            return;
        }
        const size_t fixedIdx = idx % (H_out * W_out * C * K_h * K_w);
        const size_t currentBatch = idx / (H_out * W_out * C * K_h * K_w);
        const size_t k = fixedIdx / (C * K_h * K_w);
        const size_t m = fixedIdx % (C * K_h * K_w);
        const size_t c = m / (K_h * K_w);
        const long long h = (k / W_out) * stride - pad + (m % (K_h * K_w)) / K_w;
        const long long w = (k % W_out) * stride - pad + m % K_w;
        if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
            atomicAdd(out + currentBatch * (C * H_in * W_in) + c * (H_in * W_in) + h * W_in + w, in[idx]);
        }
    }

    void img2colBackward(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const size_t H_out,
                         const size_t W_out, const size_t C, const size_t K_h, const size_t K_w, const size_t stride,
                         const size_t pad, const size_t H_in, const size_t W_in, const size_t batch) {
        StreamManager<float>::Instance().submit(img2colBackwardKernel, gridDim, blockDim, 0, out, in, H_out,
                                                W_out, C, K_h, K_w, stride, pad, H_in, W_in, batch);
    }

    __global__ void col2imgKernel(float* out, const float* in, const size_t H_out, const size_t W_out,
                                  const size_t C_out, const size_t batches) {
        const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= H_out * W_out * C_out * batches) {
            return;
        }
        const size_t batch = idx / (C_out * H_out * W_out);
        const size_t fixedIdx = idx % (C_out * H_out * W_out);
        const size_t c = fixedIdx / (H_out * W_out);
        const size_t h = (fixedIdx % (H_out * W_out)) / W_out;
        const size_t w = (fixedIdx % (H_out * W_out)) % W_out;
        out[idx] = in[batch * (C_out * H_out * W_out) + (h * W_out + w) * C_out + c];
    }

    void col2img(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const size_t H_out,
                 const size_t W_out, const size_t C_out, const size_t batches) {
        StreamManager<float>::Instance().submit(col2imgKernel, gridDim, blockDim, 0, out, in, H_out, W_out, C_out,
                                                batches);
    }

    __global__ void col2imgBackwardKernel(float* out, const float* in, const size_t H_out, const size_t W_out,
                                          const size_t C_out, const size_t batches) {
        const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= H_out * W_out * C_out * batches) {
            return;
        }
        const size_t batch = idx / (C_out * H_out * W_out);
        const size_t fixedIdx = idx % (C_out * H_out * W_out);
        const size_t c = fixedIdx / (H_out * W_out);
        const size_t h = (fixedIdx % (H_out * W_out)) / W_out;
        const size_t w = (fixedIdx % (H_out * W_out)) % W_out;
        out[batch * (C_out * H_out * W_out) + (h * W_out + w) * C_out + c] = in[idx];
    }

    void col2imgBackward(const dim3 gridDim, const dim3 blockDim, float* out, float* in, const size_t H_out,
                         const size_t W_out, const size_t C_out, const size_t batches) {
        StreamManager<float>::Instance().submit(col2imgBackwardKernel, gridDim, blockDim, 0, out, in, H_out, W_out,
                                                C_out, batches);
    }

    __global__ void AveragePoolingKernel(float* out, const float* in, const size_t pool_size, const size_t stride, const size_t padding,
        const size_t batches, const size_t channels, const size_t H_in, const size_t W_in, const size_t H_out, const size_t W_out) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batches * channels * H_out * W_out) {
            return;
        }
        const size_t currentBatch = idx / (channels * H_out * W_out);
        const size_t currentChannel = (idx % (channels * H_out * W_out)) / (H_out * W_out);
        const size_t h = (idx % (H_out * W_out)) / W_out;
        const size_t w = (idx % (H_out * W_out)) % W_out;
        const long long h_start = h * stride - padding;
        const long long w_start = w * stride - padding;
        out[idx] = 0.0f;
        size_t count = 0;
        for (long long i = 0; i < pool_size; i++) {
            for (long long j = 0; j < pool_size; j++) {
                const long long h_in = h_start + i;
                const long long w_in = w_start + j;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    out[idx] += in[currentBatch * (channels * H_in * W_in) + currentChannel * (H_in * W_in) + h_in * W_in + w_in];
                    count++;
                }
            }
        }
        out[idx] = count > 0 ? out[idx] / (float)count : 0.0f;
    }

    void AveragePooling(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
        const size_t pool_size, const size_t stride, const size_t padding,
        const size_t batches, const size_t channels, const size_t H_in, const size_t W_in,
        const size_t H_out, const size_t W_out) {
        StreamManager<float>::Instance().submit(AveragePoolingKernel, gridDim, blockDim, 0, out, in,
            pool_size, stride, padding, batches, channels, H_in, W_in, H_out, W_out);
    }

    __global__ void AveragePoolingBackwardKernel(float* out, const float* in, const size_t pool_size, const size_t stride, const size_t padding,
    const size_t batches, const size_t channels, const size_t H_in, const size_t W_in, const size_t H_out, const size_t W_out) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batches * channels * H_out * W_out) {
            return;
        }
        const size_t currentBatch = idx / (channels * H_out * W_out);
        const size_t currentChannel = (idx % (channels * H_out * W_out)) / (H_out * W_out);
        const size_t h = (idx % (H_out * W_out)) / W_out;
        const size_t w = (idx % (H_out * W_out)) % W_out;
        const long long h_start = h * stride - padding;
        const long long w_start = w * stride - padding;
        if (!padding) {
            for (long long i = 0; i < pool_size; i++) {
                for (long long j = 0; j < pool_size; j++) {
                    const long long h_in = h_start + i;
                    const long long w_in = w_start + j;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        atomicAdd(out + currentBatch * (channels * H_in * W_in) + currentChannel * (H_in * W_in) + h_in * W_in + w_in, in[idx] / (float)(pool_size*pool_size));
                    }
                }
            }
        } else {
            size_t count = 0;
            for (long long i = 0; i < pool_size; i++) {
                for (long long j = 0; j < pool_size; j++) {
                    const long long h_in = h_start + i;
                    const long long w_in = w_start + j;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        count++;
                    }
                }
            }
            for (long long i = 0; i < pool_size; i++) {
                for (long long j = 0; j < pool_size; j++) {
                    const long long h_in = h_start + i;
                    const long long w_in = w_start + j;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        atomicAdd(out + currentBatch * (channels * H_in * W_in) + currentChannel * (H_in * W_in) + h_in * W_in + w_in, in[idx] / (float)count);
                    }
                }
            }
        }
    }

    void AveragePoolingBackward(const dim3 gridDim, const dim3 blockDim, float* out, float* in,
        const size_t pool_size, const size_t stride, const size_t padding,
        const size_t batches, const size_t channels, const size_t H_in, const size_t W_in,
        const size_t H_out, const size_t W_out) {
        StreamManager<float>::Instance().submit(AveragePoolingBackwardKernel, gridDim, blockDim, 0, out, in,
            pool_size, stride, padding, batches, channels, H_in, W_in, H_out, W_out);
    }
}

//
// Created by Administrator on 24-11-11.
//

#include "NeuZephyr/OperationKernels.cuh"

namespace NeuZephyr::Operator {
    __global__ void add_kernel(const float* a, const float* b, float* c,
                               unsigned long long n) {
        unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }

    __global__ void sub_kernel(const float* a, const float* b, float* c,
                               unsigned long long n) {
        unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] - b[idx];
        }
    }

    __global__ void GEMM_kernel(const float* A, const float* B, float* C,
                                const unsigned long long M,
                                const unsigned long long N,
                                const unsigned long long K) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        const unsigned long long row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned long long col = blockIdx.x * blockDim.x + threadIdx.x;

        float sum = 0.0f;

        for (int t = 0; t < K; t += TILE_SIZE) {
            if (row < M && t + threadIdx.x < K)
                As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = 0.0f;

            if (col < N && t + threadIdx.y < K)
                Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
            else
                Bs[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();

            for (int i = 0; i < TILE_SIZE; i++)
                sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

            __syncthreads();
        }

        if (row < M && col < N)
            C[row * N + col] = sum;
    }

    __global__ void Transpose_kernel(const float* d_A, float* d_B,
                                     const unsigned int rows,
                                     const unsigned int cols) {
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

    __global__ void ScalarMul_kernel(float* out, const float* in, const float num,
                                     unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] * num;
        }
    }

    __global__ void ScalarDiv_kernel(float* out, const float* in, const float num,
                                     unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] / num;
        }
    }

    __global__ void ScalarAdd_kernel(float* out, const float* in, const float num,
                                     unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] + num;
        }
    }

    __global__ void Negation_kernel(float* out, const float* in,
                                    unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = -in[idx];
        }
    }

    __global__ void
    Recip_kernel(float* out, const float* in, unsigned long long n) {
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

    __global__ void ReLU_kernel(float* out, const float* in, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] > 0 ? in[idx] : 0;
        }
    }

    __global__ void ReLUBackward_kernel(float* A_grad, const float* A,
                                        const float* B_grad, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = A[idx] > 0 ? B_grad[idx] : 0;
        }
    }

    __global__ void Sigmoid_kernel(float* out, const float* in,
                                   unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = 1.0f / (1.0f + __expf(-in[idx]));
        }
    }

    __global__ void SigmoidBackward_kernel(float* A_grad, const float* B,
                                           const float* B_grad,
                                           unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = B[idx] * (1.0f - B[idx]) * B_grad[idx];
        }
    }

    __global__ void Tanh_kernel(float* out, const float* in, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = __tanf(in[idx]);
        }
    }

    __global__ void TanhBackward_kernel(float* A_grad, const float* B,
                                        const float* B_grad, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = (1.0f - B[idx] * B[idx]) * B_grad[idx];
        }
    }

    __global__ void LeakyReLU_kernel(float* out, const float* in,
                                     unsigned long long n, float alpha) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] > 0 ? in[idx] : alpha * in[idx];
        }
    }

    __global__ void LeakyReLUBackward_kernel(float* A_grad, const float* A,
                                             const float* B_grad,
                                             unsigned long long n,
                                             float alpha) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = A[idx] > 0 ? B_grad[idx] : alpha * B_grad[idx];
        }
    }

    __global__ void
    Swish_kernel(float* out, const float* in, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] / (1.0f + __expf(-in[idx]));
        }
    }

    __global__ void SwishBackward_kernel(float* A_grad, const float* A,
                                         const float* B, const float* B_grad,
                                         unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = 1.0f / (1.0f + __expf(-A[idx])) + B[idx] * (1.0f - B[idx]) *
                B_grad[idx];
        }
    }

    __global__ void ELU_kernel(float* out, const float* in, unsigned long long n,
                               float alpha) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] > 0 ? in[idx] : alpha * (__expf(in[idx]) - 1);
        }
    }

    __global__ void ELUBackward_kernel(float* A_grad, const float* A,
                                       const float* B_grad, unsigned long long n,
                                       float alpha) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = A[idx] > 0
                              ? B_grad[idx]
                              : alpha * __expf(A[idx]) * B_grad[idx];
        }
    }

    __global__ void HardSigmoid_kernel(float* out, const float* in,
                                       unsigned long long n, float alpha,
                                       float beta) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] * alpha + beta;
            out[idx] = out[idx] > 1.0f ? 1.0f : (out[idx] < 0.0f ? 0.0f : out[idx]);
        }
    }

    __global__ void HardSigmoidBackward_kernel(float* A_grad, const float* A,
                                               const float* B_grad,
                                               unsigned long long n,
                                               float alpha, float beta) {
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

    __inline__ __device__ float LiteHardSigmoid(float x, float alpha, float beta) {
        float a = x * alpha + beta;
        return a > 1.0f ? 1.0f : (a < 0.0f ? 0.0f : a);
    }

    __global__ void HardSwish_kernel(float* out, const float* in,
                                     unsigned long long n, float alpha,
                                     float beta) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx] * LiteHardSigmoid(in[idx], alpha, beta);
        }
    }

    __global__ void HardSwishBackward_kernel(float* A_grad, const float* A,
                                             const float* B_grad,
                                             unsigned long long n,
                                             float alpha, float beta) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            A_grad[idx] = LiteHardSigmoid(A[idx], alpha, beta) + B_grad[idx] * A[idx] *
                alpha * (1 - LiteHardSigmoid(
                    A[idx], alpha, beta));
        }
    }

    __global__ void ExpSum_kernel(float* out, const float* g_data,
                                  unsigned long long n) {
        extern __shared__ float sdata[];
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long tid = threadIdx.x;

        // copy data
        if (idx < n) {
            sdata[tid] = __expf(g_data[idx]);
        }
        else {
            sdata[tid] = 0;
        }
        __syncthreads();

        // loop unroll
        if (blockDim.x >= 1024 && tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }

        __syncthreads();

        if (blockDim.x >= 512 && tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }

        __syncthreads();

        if (blockDim.x >= 256 && tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }

        __syncthreads();

        if (blockDim.x >= 128 && tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }

        __syncthreads();

        // warp unroll
        if (tid < 32) {
            volatile float* vdata = sdata;
            vdata[tid] += vdata[tid + 32];
            vdata[tid] += vdata[tid + 16];
            vdata[tid] += vdata[tid + 8];
            vdata[tid] += vdata[tid + 4];
            vdata[tid] += vdata[tid + 2];
            vdata[tid] += vdata[tid + 1];
        }

        __syncthreads();

        // write data back to global memory
        if (tid == 0) {
            out[blockIdx.x] = sdata[0];
        }
    }

    __global__ void Softmax_kernel(float* out, const float* in,
                                   float exp_sum_of_input, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = __expf(in[idx]) / exp_sum_of_input;
        }
    }

    __global__ void SoftmaxJacobian_kernel(float* out, const float* in,
                                           unsigned long long n) {
        unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long long idy = blockIdx.y * blockDim.y + threadIdx.y;
        if (idx >= n || idy >= n) {
            return;
        }
        unsigned long long id = idx * n + idy;
        if (idy == idx) {
            out[id] = in[idx] * (1 - in[idx]);
        }
        else {
            out[id] = -in[idx] * in[idy];
        }
    }

    __global__ void MSE_kernel(float* out, const float* predict, const float* real,
                               unsigned long long n) {
        extern __shared__ float smem[];
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long tid = threadIdx.x;
        if (idx < n) {
            smem[tid] = (predict[idx] - real[idx]) * (predict[idx] - real[idx]) / (
                float)n;
        }
        else {
            smem[tid] = 0;
        }
        __syncthreads();
        // loop unroll
        if (blockDim.x >= 1024 && tid < 512) {
            smem[tid] += smem[tid + 512];
        }
        __syncthreads();
        if (blockDim.x >= 512 && tid < 256) {
            smem[tid] += smem[tid + 256];
        }
        __syncthreads();
        if (blockDim.x >= 256 && tid < 128) {
            smem[tid] += smem[tid + 128];
        }
        __syncthreads();
        if (blockDim.x >= 128 && tid < 64) {
            smem[tid] += smem[tid + 64];
        }
        __syncthreads();
        // warp unroll
        if (tid < 32) {
            volatile float* vdata = smem;
            vdata[tid] += vdata[tid + 32];
            vdata[tid] += vdata[tid + 16];
            vdata[tid] += vdata[tid + 8];
            vdata[tid] += vdata[tid + 4];
            vdata[tid] += vdata[tid + 2];
            vdata[tid] += vdata[tid + 1];
        }
        __syncthreads();
        // write data back to global memory
        if (tid == 0) {
            out[blockIdx.x] = smem[0];
        }
    }

    __global__ void MSEBackward_kernel(float* out, const float* predict,
                                       const float* real, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = 2 * (predict[idx] - real[idx]) / (float)n;
        }
    }

    __global__ void SGD_kernel(float* data, const float* grad, const float lr,
                               unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] -= lr * grad[idx];
        }
    }

    __global__ void BCE_kernel(float* out, const float* predict, const float* real,
                               unsigned long long n) {
        extern __shared__ float smem[];
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long tid = threadIdx.x;
        if (idx < n) {
            smem[tid] = (-real[idx] * __logf(predict[idx]) - (1 - real[idx]) *
                __logf(1 - predict[idx])) / (float)n;
        }
        else {
            smem[tid] = 0;
        }
        __syncthreads();
        // loop unroll
        if (blockDim.x >= 1024 && tid < 512) {
            smem[tid] += smem[tid + 512];
        }
        __syncthreads();
        if (blockDim.x >= 512 && tid < 256) {
            smem[tid] += smem[tid + 256];
        }
        __syncthreads();
        if (blockDim.x >= 256 && tid < 128) {
            smem[tid] += smem[tid + 128];
        }
        __syncthreads();
        if (blockDim.x >= 128 && tid < 64) {
            smem[tid] += smem[tid + 64];
        }
        __syncthreads();

        // warp unroll
        if (tid < 32) {
            volatile float* vdata = smem;
            vdata[tid] += vdata[tid + 32];
            vdata[tid] += vdata[tid + 16];
            vdata[tid] += vdata[tid + 8];
            vdata[tid] += vdata[tid + 4];
            vdata[tid] += vdata[tid + 2];
            vdata[tid] += vdata[tid + 1];
        }
        __syncthreads();
        // write data back to global memory
        if (tid == 0) {
            out[blockIdx.x] = smem[0];
        }
    }

    __global__ void BCEBackward_kernel(float* out, const float* predict,
                                       const float* real, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = ((predict[idx] - real[idx]) / (
                predict[idx] * (1 - predict[idx]))) / (float)n;
        }
    }

    __global__ void Momentum_kernel(float* output, const float* grad,
                                    const float* velocity, float beta,
                                    unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = velocity[idx] * beta + grad[idx] * (1 - beta);
        }
    }

    __global__ void SquaredSum_kernel(float* output, const float* input, unsigned long long n) {
        extern __shared__ float s_data[];
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long tid = threadIdx.x;
        if (idx < n) {
            s_data[tid] = input[idx] * input[idx];
        } else {
            s_data[tid] = 0;
        }
        __syncthreads();
        // loop unroll
        if (blockDim.x >= 1024 && tid < 512) {
            s_data[tid] += s_data[tid + 512];
        }
        __syncthreads();
        if (blockDim.x >= 512 && tid < 256) {
            s_data[tid] += s_data[tid + 256];
        }
        __syncthreads();
        if (blockDim.x >= 256 && tid < 128) {
            s_data[tid] += s_data[tid + 128];
        }
        __syncthreads();
        if (blockDim.x >= 128 && tid < 64) {
            s_data[tid] += s_data[tid + 64];
        }
        __syncthreads();
        if (tid < 32) {
            volatile float* vdata = s_data;
            vdata[tid] += vdata[tid + 32];
            vdata[tid] += vdata[tid + 16];
            vdata[tid] += vdata[tid + 8];
            vdata[tid] += vdata[tid + 4];
            vdata[tid] += vdata[tid + 2];
            vdata[tid] += vdata[tid + 1];
        }
        __syncthreads();
        if (tid == 0) {
            output[blockIdx.x] = s_data[0];
        }
    }

    __global__ void Ada_kernel(float* data, const float* grad, const float G, const float lr, const float theta, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            const float current_lr = lr / sqrtf(G + theta);
            data[idx] -= current_lr * grad[idx];
        }
    }
}

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

    __global__ void AdaGrad_kernel(float* data, float* G, const float* grad, const float lr, const float eps,
                                   unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float temp = G[idx] + grad[idx] * grad[idx];
        data[idx] -= lr * grad[idx] / (sqrtf(temp) + eps);
        G[idx] = temp;
    }

    __global__ void RMSprop_kernel(float* data, float* v, const float* grad, const float lr, const float beta,
                                   const float eps, unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float temp = v[idx] * beta + grad[idx] * grad[idx] * (1 - beta);
        data[idx] -= lr * grad[idx] / (sqrtf(temp) + eps);
        v[idx] = temp;
    }

    __global__ void Adam_kernel(float* data, float* m, float* v, const float* grad, const float lr, const float beta1,
                                const float beta2, const float eps, const int t, unsigned long long n) {
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

    __global__ void NAdam_kernel(float* data, float* m, float* m_modified, float* v, const float* grad, const float lr,
                                 const float beta1, const float beta2, const float eps, const int t,
                                 unsigned long long n) {
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

    __global__ void AdaDelta_kernel(float* data, float* acc_delta, float* acc_grad, const float* grad,
                                    const float rho, const float eps,
                                    unsigned long long n) {
        const unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float delta_acc_grad_temp = acc_grad[idx] * rho + grad[idx] * grad[idx] * (1 - rho);
        const float delta_theta = - grad[idx] * sqrtf(acc_delta[idx] + eps) / sqrtf(delta_acc_grad_temp + eps);
        data[idx] += delta_theta;
        const float delta_acc_temp = acc_delta[idx] * rho + delta_theta * delta_theta * (1 - rho);
        acc_delta[idx] = delta_acc_temp;
        acc_grad[idx] = delta_acc_grad_temp;
    }
}

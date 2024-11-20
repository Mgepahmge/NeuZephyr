//
// Created by Administrator on 24-11-20.
//

#include "NeuZephyr/Optimizer.cuh"

namespace NeuZephyr::Optimizers {
    SGD::SGD(const Tensor::value_type learning_rate) {
        this->learning_rate = learning_rate;
    }

    void SGD::step(Node *input) {
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        SGD_kernel<<<grid, block>>>(input->output->data(), input->output->grad(), learning_rate, input->output->size());
    }

    Momentum::Momentum(Tensor::value_type learning_rate, Tensor::value_type beta) {
        this->learning_rate = learning_rate;
        this->beta = beta;
    }

    void Momentum::step(Node *input) {
        if (velocity.find(input) == velocity.end()) {
            Tensor v(input->output->shape(), false);
            v.fill(0);
            velocity[input] = v;
        }
        float* temp;
        cudaMalloc(&temp, input->output->size() * sizeof(float));
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        Momentum_kernel<<<grid, block>>>(temp, input->output->grad(), velocity[input].data(), beta, input->output->size());
        cudaMemcpy(velocity[input].data(), temp, input->output->size() * sizeof(float), cudaMemcpyDeviceToDevice);
        SGD_kernel<<<grid, block>>>(input->output->data(), velocity[input].data(), learning_rate, input->output->size());
        cudaFree(temp);
    }

    Ada::Ada(Tensor::value_type learning_rate) {
        this->learning_rate = learning_rate;
    }

    void Ada::step(Node* input) {
        if (G.find(input) == G.end()) {
            G[input] = 0;
        }
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        float* temp;
        float* temp_host;
        cudaMalloc(&temp, grid.x * sizeof(float));
        temp_host = static_cast<float*>(malloc(grid.x * sizeof(float)));
        SquaredSum_kernel<<<grid, block, block.x*sizeof(float)>>>(temp, input->output->grad(), input->output->size());
        cudaMemcpy(temp_host, temp, grid.x*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid.x; i++) {
            G[input] += temp_host[i];
        }
        cudaFree(temp);
        free(temp_host);
        Ada_kernel<<<grid, block>>>(input->output->data(), input->output->grad(), G[input], learning_rate, theta, input->output->size());
    }
} // Optimizers
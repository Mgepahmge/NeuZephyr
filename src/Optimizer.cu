//
// Created by Administrator on 24-11-20.
//

#include "NeuZephyr/Optimizer.cuh"

namespace NeuZephyr::Optimizers {
    SGD::SGD(const Tensor::value_type learning_rate) {
        this->learning_rate = learning_rate;
    }

    void SGD::step(Node* input) {
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        SGD_kernel<<<grid, block>>>(input->output->data(), input->output->grad(), learning_rate, input->output->size());
    }

    Momentum::Momentum(Tensor::value_type learning_rate, Tensor::value_type beta) {
        this->learning_rate = learning_rate;
        this->beta = beta;
    }

    void Momentum::step(Node* input) {
        if (velocity.find(input) == velocity.end()) {
            Tensor v(input->output->shape(), false);
            v.fill(0);
            velocity[input] = v;
        }
        float* temp;
        cudaMalloc(&temp, input->output->size() * sizeof(float));
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        Momentum_kernel<<<grid, block>>>(temp, input->output->grad(), velocity[input].data(), beta,
                                         input->output->size());
        cudaMemcpy(velocity[input].data(), temp, input->output->size() * sizeof(float), cudaMemcpyDeviceToDevice);
        SGD_kernel<<<grid, block>>>(input->output->data(), velocity[input].data(), learning_rate,
                                    input->output->size());
        cudaFree(temp);
    }

    AdaGrad::AdaGrad(Tensor::value_type learning_rate) {
        this->learning_rate = learning_rate;
    }

    void AdaGrad::step(Node* input) {
        if (G.find(input) == G.end()) {
            Tensor g(input->output->shape(), false);
            g.fill(0);
            G[input] = g;
        }
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        AdaGrad_kernel<<<grid, block>>>(input->output->data(),  G[input].data(), input->output->grad(),learning_rate, epsilon, input->output->size());
    }
} // Optimizers

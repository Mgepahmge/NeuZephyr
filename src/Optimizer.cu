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
} // Optimizers
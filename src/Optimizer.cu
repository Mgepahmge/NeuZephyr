#include "NeuZephyr/Optimizer.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/StreamManager.cuh"
#include <fstream>

using namespace nz::krnl;


namespace nz::opt {
    SGD::SGD(const Tensor::value_type learning_rate) {
        this->learning_rate = learning_rate;
    }

    void SGD::step(Node* input) {
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        StochasticGradientDescent(grid, block, input->output->data(), input->output->grad(), learning_rate,
                                  input->output->size());
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
        cuStrm::StreamManager<Tensor::value_type>::Instance().malloc(&temp, input->output->size() * sizeof(float));
        const dim3 block(256);
        const dim3 grid((input->output->size() + block.x - 1) / block.x);
        krnl::Momentum(grid, block, temp, input->output->grad(), velocity[input].data(), beta,
                       input->output->size());
        cuStrm::StreamManager<Tensor::value_type>::Instance().memcpy(velocity[input].data(), temp,
                                                                     input->output->size() * sizeof(float),
                                                                     cudaMemcpyDeviceToDevice);
        StochasticGradientDescent(grid, block, input->output->data(), velocity[input].data(), learning_rate,
                                  input->output->size());
        cuStrm::StreamManager<Tensor::value_type>::Instance().free(temp);
    }

    AdaGrad::AdaGrad(Tensor::value_type learning_rate) {
        this->learning_rate = learning_rate;
    }

    void AdaGrad::step(Node* input) {
        if (gss.find(input) == gss.end()) {
            Tensor g(input->output->shape(), false);
            g.fill(0);
            gss[input] = g;
        }
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        krnl::AdaGrad(grid, block, input->output->data(), gss[input].data(), input->output->grad(),
                      learning_rate, epsilon, input->output->size());
    }

    RMSprop::RMSprop(Tensor::value_type learning_rate, Tensor::value_type decay_rate) {
        this->learning_rate = learning_rate;
        this->decay_rate = decay_rate;
    }

    void RMSprop::step(Node* input) {
        if (v.find(input) == v.end()) {
            Tensor v_(input->output->shape(), false);
            v_.fill(0);
            v[input] = v_;
        }
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        krnl::RMSprop(grid, block, input->output->data(), v[input].data(), input->output->grad(), learning_rate,
                      decay_rate, epsilon, input->output->size());
    }

    Adam::Adam(Tensor::value_type learning_rate, Tensor::value_type beta1, Tensor::value_type beta2) {
        this->learning_rate = learning_rate;
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->it = 0;
    }

    void Adam::step(Node* input) {
        it++;
        if (m.find(input) == m.end()) {
            Tensor m_(input->output->shape(), false);
            m_.fill(0);
            m[input] = m_;
        }
        if (v.find(input) == v.end()) {
            Tensor v_(input->output->shape(), false);
            v_.fill(0);
            v[input] = v_;
        }
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        krnl::Adam(grid, block, input->output->data(), m[input].data(), v[input].data(), input->output->grad(),
                   learning_rate, beta1, beta2, epsilon, it, input->output->size());
    }

    NAdam::NAdam(Tensor::value_type learning_rate, Tensor::value_type beta1, Tensor::value_type beta2) {
        this->learning_rate = learning_rate;
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->it = 0;
    }

    void NAdam::step(Node* input) {
        it++;
        if (m.find(input) == m.end()) {
            Tensor m_(input->output->shape(), false);
            m_.fill(0);
            m[input] = m_;
        }
        if (v.find(input) == v.end()) {
            Tensor v_(input->output->shape(), false);
            v_.fill(0);
            v[input] = v_;
        }
        if (m_modified.find(input) == m_modified.end()) {
            Tensor m_mod_(input->output->shape(), false);
            m_mod_.fill(0);
            m_modified[input] = m_mod_;
        }
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        krnl::NAdam(grid, block, input->output->data(), m[input].data(), m_modified[input].data(),
                    v[input].data(),
                    input->output->grad(), learning_rate, beta1, beta2, epsilon, it,
                    input->output->size());
    }

    AdaDelta::AdaDelta(Tensor::value_type rho) {
        this->learning_rate = rho;
    }

    void AdaDelta::step(Node* input) {
        if (acc_delta.find(input) == acc_delta.end()) {
            Tensor delta_acc_(input->output->shape(), false);
            delta_acc_.fill(0);
            acc_delta[input] = delta_acc_;
        }
        if (acc_grad.find(input) == acc_grad.end()) {
            Tensor delta_acc_grad_(input->output->shape(), false);
            delta_acc_grad_.fill(0);
            acc_grad[input] = delta_acc_grad_;
        }
        dim3 block(256);
        dim3 grid((input->output->size() + block.x - 1) / block.x);
        krnl::AdaDelta(grid, block, input->output->data(), acc_delta[input].data(), acc_grad[input].data(),
                       input->output->grad(), learning_rate, epsilon,
                       input->output->size());
    }
} // opt

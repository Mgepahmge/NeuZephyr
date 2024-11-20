//
// Created by Administrator on 24-11-20.
//

#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH
#include <unordered_map>

#include "Nodes.cuh"
#include "OperationKernels.cuh"
#include "Tensor.cuh"

namespace NeuZephyr::Optimizers {
    using namespace data;
    using namespace Operator;
    using namespace Nodes;

    class DL_API Optimizer {
    protected:
        Tensor::value_type learning_rate;

    public:
        explicit Optimizer() = default;

        virtual ~Optimizer() = default;

        virtual void step(Node* input) = 0;
    };

    class DL_API SGD : public Optimizer {
    public:
        explicit SGD(Tensor::value_type learning_rate);
        void step(Node* input) override;
    };

    class DL_API Momentum : public Optimizer {
        std::pmr::unordered_map<Node*, Tensor> velocity;
        Tensor::value_type beta;

    public:
        explicit Momentum(Tensor::value_type learning_rate, Tensor::value_type beta);
        void step(Node* input) override;
    };

    class DL_API Ada : public Optimizer {
        std::unordered_map<Node*, Tensor::value_type> G;
        Tensor::value_type theta = 1e-8;
    public:
        explicit Ada(Tensor::value_type learning_rate);
        void step(Node* input) override;
    };
} // namespace NeuZephyr::Optimizers

#endif // OPTIMIZER_CUH

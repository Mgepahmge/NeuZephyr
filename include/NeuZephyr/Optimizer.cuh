//
// Created by Administrator on 24-11-20.
//

#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH
#include "Tensor.cuh"
#include "OperationKernels.cuh"
#include "Nodes.cuh"

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

} // Optimizers

#endif //OPTIMIZER_CUH

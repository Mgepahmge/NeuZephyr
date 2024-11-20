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
class Optimizer {
public:
    explicit Optimizer();

    virtual ~Optimizer() = default;

    virtual void step(Node* input) = 0;
};

} // Optimizers

#endif //OPTIMIZER_CUH

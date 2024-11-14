//
// Created by Administrator on 24-11-11.
//

#ifndef NODES_CUH
#define NODES_CUH

#include "Tensor.cuh"

namespace NeuZephyr::Nodes {
    class Node {
    public:
        virtual ~Node() = default;

        std::vector<Node*> inputs;
        std::shared_ptr<data::Tensor> output;
        virtual void forward() = 0;
        virtual void backward() = 0;
    };
}

#endif //NODES_CUH

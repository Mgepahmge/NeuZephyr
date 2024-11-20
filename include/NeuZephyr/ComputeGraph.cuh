//
// Created by Administrator on 24-11-20.
//

#ifndef COMPUTEGRAPH_CUH
#define COMPUTEGRAPH_CUH

#include "Nodes.cuh"
#include <unordered_map>
#include <string>
#include <queue>

namespace NeuZephyr::Graph {
    using namespace Nodes;
    using namespace data;
    using namespace Operator;

    class DL_API ComputeGraph {
    public:
        std::vector<Node*> nodes;
        std::vector<Node*> input_nodes;
        std::vector<Node*> output_nodes;
        std::vector<Node*> sorted_nodes;

        std::unordered_map<Node*, int> in_degree;
        std::unordered_map<Node*, std::vector<Node*>> adj_list;
        std::unordered_map<std::string, Node*> node_roster;
        std::unordered_map<Node*, std::string> node_roster_reverse;
        int nodes_ref = 0;

    public:
        ComputeGraph() = default;
        ~ComputeGraph();
        std::ostream& print(std::ostream& os);
        friend DL_API std::ostream& operator<<(std::ostream& os, ComputeGraph& graph);
        void topological_sort();

        InputNode* add_input(const Tensor::shape_type &shape, bool requires_grad = false, const std::string& name = "default");
        InputNode* add_input(const Tensor& tensor, const std::string& name = "default");
        InputNode* add_input(const std::initializer_list<int>& shape, bool requires_grad = false, const std::string& name = "default");
    };

}

#endif //COMPUTEGRAPH_CUH

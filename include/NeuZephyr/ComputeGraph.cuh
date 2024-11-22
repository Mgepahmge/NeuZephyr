//
// Created by Administrator on 24-11-20.
//

#ifndef COMPUTEGRAPH_CUH
#define COMPUTEGRAPH_CUH

#include "Nodes.cuh"
#include <unordered_map>
#include <string>
#include <queue>
#include "Optimizer.cuh"

namespace NeuZephyr::Graph {
    using namespace Nodes;
    using namespace data;
    using namespace Operator;
    using namespace Optimizers;

    class DL_API ComputeGraph {
        std::vector<Node*> nodes;
        std::vector<Node*> input_nodes;
        std::vector<OutputNode*> output_nodes;
        std::vector<Node*> sorted_nodes;

        std::unordered_map<Node*, int> in_degree;
        std::unordered_map<Node*, std::vector<Node*>> adj_list;
        std::unordered_map<std::string, Node*> node_roster;
        std::unordered_map<Node*, std::string> node_roster_reverse;
        int nodes_ref = 0;

    public:
        ComputeGraph() = default;
        ~ComputeGraph() = default;
        std::ostream& print(std::ostream& os);
        friend DL_API std::ostream& operator<<(std::ostream& os, ComputeGraph& graph);
        void topological_sort();

        InputNode* add_input(const Tensor::shape_type &shape, bool requires_grad = false, const std::string& name = "default");
        InputNode* add_input(const Tensor& tensor, const std::string& name = "default");
        InputNode* add_input(const std::initializer_list<int>& shape, bool requires_grad = false, const std::string& name = "default");
        InputNode* add_input(InputNode* input, const std::string& name);
        template<typename NodeType>
        NodeType* add_node(NodeType* node, const std::string& name = "default") {
            nodes.push_back(node);
            if (name == "default") {
            const std::string node_name = node->type + "_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
            } else {
            node_roster[name] = node;
            node_roster_reverse[node] = name;
            }
            return node;
        }
        OutputNode* add_output(OutputNode* node, const std::string& name = "default");
        void forward();
        void backward();
        void zero_grad() const;
        void randomize(const std::string& name);
        void randomize(const Node* node);
        void randomize_all() const;
        void fill(const std::string& name, Tensor::value_type val);
        void fill(const Node* node, Tensor::value_type val);
        void fill_all(Tensor::value_type val) const;
        void set_input(const std::string& name, const Tensor::value_type* data);
        void set_input(const Node* node, const Tensor::value_type* data);
        Tensor::value_type* get_output() const;
        Tensor::value_type* get_output_host() const;
        OutputNode* get_output_node() const;
        Tensor::value_type get_loss() const;
        void update(Optimizer* optimizer) const;
    };

}

#endif //COMPUTEGRAPH_CUH

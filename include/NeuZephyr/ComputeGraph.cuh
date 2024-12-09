//
// Created by Administrator on 24-11-20.
//

#ifndef COMPUTEGRAPH_CUH
#define COMPUTEGRAPH_CUH

#include "Nodes.cuh"
#include <unordered_map>
#include <string>
#include <queue>
#include <fstream>
#include <sstream>
#include "Optimizer.cuh"

namespace NeuZephyr::Graph {
    using namespace Nodes;
    using namespace Data;
    using namespace Kernels;
    using namespace Optimizers;
    using namespace Nodes::Standard;
    using namespace Nodes::Computation;
    using namespace Nodes::Loss;

    template <typename T, typename... Args>
    std::unique_ptr<T> createInstance(Args&&... args) {
        return std::make_unique<T>(std::forward<Args>(args)...);
    }

    class DL_API ComputeGraph {
        std::vector<Node*> nodes;
        std::vector<Node*> inputNodes;
        std::vector<OutputNode*> outputNodes;
        std::vector<Node*> sortedNodes;

        std::unordered_map<Node*, int> inDegree;
        std::unordered_map<Node*, std::vector<Node*>> adjList;
        std::unordered_map<std::string, Node*> nodeRoster;
        std::unordered_map<Node*, std::string> nodeRosterReverse;
        int nodesRef = 0;

    public:
        ComputeGraph() = default;
        ~ComputeGraph() = default;
        std::ostream& print(std::ostream& os);
        friend DL_API std::ostream& operator<<(std::ostream& os, ComputeGraph& graph);
        friend DL_API void CreateNode(ComputeGraph* graph, const std::string& type, const std::string& name,
                                      std::vector<int> pre,
                                      const std::vector<int>& shape, const float* data, bool requires_grad,
                                      const float* grad);
        void topologicalSort();

        InputNode* addInput(const Tensor::shape_type& shape, bool requires_grad = false,
                            const std::string& name = "default");
        InputNode* addInput(const Tensor& tensor, const std::string& name = "default");
        InputNode* addInput(const std::initializer_list<int>& shape, bool requires_grad = false,
                            const std::string& name = "default");
        InputNode* addInput(InputNode* input, const std::string& name);

        template <typename NodeType>
        NodeType* addNode(NodeType* node, const std::string& name = "default") {
            nodes.push_back(node);
            if (name == "default") {
                const std::string node_name = node->type + "_" + std::to_string(nodesRef);
                nodeRoster[node_name] = node;
                nodeRosterReverse[node] = node_name;
                nodesRef++;
            }
            else {
                nodeRoster[name] = node;
                nodeRosterReverse[node] = name;
            }
            return node;
        }

        OutputNode* addOutput(OutputNode* node, const std::string& name = "default");
        void forward();
        void backward();
        void zeroGrad() const;
        void randomize(const std::string& name);
        void randomize(const Node* node);
        void randomizeAll() const;
        void fill(const std::string& name, Tensor::value_type val);
        void fill(const Node* node, Tensor::value_type val);
        void fillAll(Tensor::value_type val) const;
        void setInput(const std::string& name, const Tensor::value_type* data);
        void setInput(const Node* node, const Tensor::value_type* data);
        Tensor::value_type* getOutput() const;
        Tensor::value_type* getOutputHost() const;
        OutputNode* getOutputNode() const;
        Tensor::value_type getLoss() const;
        void update(Optimizer* optimizer) const;
        void save(const std::string& path);
        void load(const std::string& path);
        Node* operator[](const std::string& name);
    };

}

#endif //COMPUTEGRAPH_CUH

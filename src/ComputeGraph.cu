//
// Created by Administrator on 24-11-20.
//

#include "NeuZephyr/ComputeGraph.cuh"

namespace NeuZephyr::Graph {
    ComputeGraph::~ComputeGraph() {
        for (auto node : nodes) {
            delete node;
        }
    }

    std::ostream & ComputeGraph::print(std::ostream &os) {
        if (sorted_nodes.empty()) {
            throw std::runtime_error("Graph is not sorted");
        }
        for (Node* node : sorted_nodes) {
            os << "Node:" << node_roster_reverse[node] << "\n";
            os << "Pre:";
            for (Node* pre : node->inputs) {
                os << " " << node_roster_reverse[pre];
            }
            os << " || ";
            os << "Next:";
            for (Node* next : adj_list[node]) {
                os << " " << node_roster_reverse[next];
            }
            os << "\n";
            os << "Data:\n";
            os << *node->output;
            os << "Grad:\n";
            node->output->print_grad(os);
            os << "\n";
        }
        return os;
    }

    std::ostream& operator<<(std::ostream &os, ComputeGraph &graph) {
        return graph.print(os);
    }

    void ComputeGraph::topological_sort() {
        sorted_nodes.clear();
        in_degree.clear();
        adj_list.clear();

        for (Node* node : nodes) {
            if (in_degree.find(node) == in_degree.end()) {
                in_degree[node] = 0;
            }

            for (Node* input : node->inputs) {
                adj_list[input].push_back(node);
                in_degree[node]++;
            }
        }

        std::queue<Node*> q;
        for (Node* node : nodes) {
            if (in_degree[node] == 0) {
                q.push(node);
            }
        }

        while(!q.empty()) {
            Node* node = q.front();
            q.pop();
            sorted_nodes.push_back(node);
            for (Node* next : adj_list[node]) {
                in_degree[next]--;
                if (in_degree[next] == 0) {
                    q.push(next);
                }
            }
        }

        if (sorted_nodes.size() != nodes.size()) {
            throw std::runtime_error("Graph has cycle");
        }
    }

    InputNode* ComputeGraph::add_input(const Tensor::shape_type &shape, bool requires_grad, const std::string& name) {
        auto node = new InputNode(shape, requires_grad);
        nodes.push_back(node);
        input_nodes.push_back(node);
        if (name == "default") {
            const std::string node_name = "input_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
        } else {
            node_roster[name] = node;
            node_roster_reverse[node] = name;
        }
        return node;
    }

    InputNode* ComputeGraph::add_input(const Tensor &tensor, const std::string& name) {
        auto node = new InputNode(tensor);
        nodes.push_back(node);
        input_nodes.push_back(node);
        if (name == "default") {
            const std::string node_name = "input_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
        } else {
            node_roster[name] = node;
            node_roster_reverse[node] = name;
        }
        return node;
    }

    InputNode* ComputeGraph::add_input(const std::initializer_list<int> &shape, bool requires_grad, const std::string& name) {
        auto node = new InputNode(shape, requires_grad);
        nodes.push_back(node);
        input_nodes.push_back(node);
        if (name == "default") {
            const std::string node_name = "input_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
        } else {
            node_roster[name] = node;
            node_roster_reverse[node] = name;
        }
        return node;
    }
} // Graph
// NeuZephyr
//
// Created by Administrator on 24-11-20.
//

#include "NeuZephyr/ComputeGraph.cuh"

namespace NeuZephyr::Graph {
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
            if (node->output->requires_grad()) {
                os << "Grad:\n";
                node->output->print_grad(os);
                os << "\n";
            }
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

    InputNode * ComputeGraph::add_input(InputNode *input, const std::string &name) {
        nodes.push_back(input);
        input_nodes.push_back(input);
        if (name == "default") {
            const std::string node_name = "input_" + std::to_string(nodes_ref);
            node_roster[node_name] = input;
            node_roster_reverse[input] = node_name;
            nodes_ref++;
        } else {
            node_roster[name] = input;
            node_roster_reverse[input] = name;
        }
        return input;
    }

    OutputNode * ComputeGraph::add_output(OutputNode *node, const std::string &name) {
        nodes.push_back(node);
        output_nodes.push_back(node);
        if (name == "default") {
            const std::string node_name = "output_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
        } else {
            node_roster[name] = node;
            node_roster_reverse[node] = name;
        }
        return node;
    }

    void ComputeGraph::forward() {
        if (sorted_nodes.empty()) {
            topological_sort();
        }
        for (Node* node : sorted_nodes) {
            node->forward();
        }
    }

    void ComputeGraph::backward() {
        if (sorted_nodes.empty()) {
            topological_sort();
        }
        if (output_nodes.size() == 1) {
            for (auto it = sorted_nodes.rbegin(); it != sorted_nodes.rend(); ++it) {
                (*it)->backward();
            }
        } else {
            if (output_nodes.empty()) {
                throw std::runtime_error("No output node");
            } else {
                throw std::runtime_error("Multiple output nodes");
            }
        }
    }

    void ComputeGraph::zero_grad() const {
        for (Node* node : nodes) {
            node->output->zero_grad();
        }
    }

    void ComputeGraph::randomize(const std::string &name) {
        node_roster[name]->output->randomize();
    }

    void ComputeGraph::randomize(const Node *node) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->randomize();
        } else {
            throw std::runtime_error("Node not in graph");
        }
    }

    void ComputeGraph::randomize_all() const {
        for (Node* node : input_nodes) {
            node->output->randomize();
        }
    }

    void ComputeGraph::fill(const std::string &name, const Tensor::value_type val) {
        node_roster[name]->output->fill(val);
    }

    void ComputeGraph::fill(const Node *node, const Tensor::value_type val) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->fill(val);
        } else {
            throw std::runtime_error("Node not in graph");
        }
    }

    void ComputeGraph::fill_all(Tensor::value_type val) const {
        for (Node* node : input_nodes) {
            node->output->fill(val);
        }
    }

    void ComputeGraph::set_input(const std::string &name, const Tensor::value_type *data) {
        node_roster[name]->output->copy_data(data, node_roster[name]->output->shape());
    }

    void ComputeGraph::set_input(const Node *node, const Tensor::value_type *data) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->copy_data(data, node->output->shape());
        } else {
            throw std::runtime_error("Node not in graph");
        }
    }

    Tensor::value_type* ComputeGraph::get_output() const {
        return output_nodes[0]->output->data();
    }

    Tensor::value_type* ComputeGraph::get_output_host() const {
        auto* data = static_cast<Tensor::value_type *>(malloc(output_nodes[0]->output->size() * sizeof(Tensor::value_type)));
        cudaMemcpy(data, output_nodes[0]->output->data(), output_nodes[0]->output->size() * sizeof(Tensor::value_type), cudaMemcpyDeviceToHost);
        return data;
    }

    OutputNode* ComputeGraph::get_output_node() const {
        return output_nodes[0];
    }

    Tensor::value_type ComputeGraph::get_loss() const {
        return output_nodes[0]->get_loss();
    }

    void ComputeGraph::update(Optimizer *optimizer) const {
        for (Node* node : nodes) {
            if (node->output->requires_grad()) {
                optimizer->step(node);
            }
        }
    }
} // Graph
// NeuZephyr
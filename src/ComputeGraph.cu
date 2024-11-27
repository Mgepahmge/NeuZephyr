//
// Created by Administrator on 24-11-20.
//

#include "NeuZephyr/ComputeGraph.cuh"

#include <filesystem>

namespace NeuZephyr::Graph {
    std::ostream& ComputeGraph::print(std::ostream& os) {
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
            os << "\n";
            if (node->output->requires_grad()) {
                os << "Grad:\n";
                node->output->print_grad(os);
                os << "\n";
            }
        }
        os << "loss: " << output_nodes[0]->get_loss() << "\n";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, ComputeGraph& graph) {
        return graph.print(os);
    }

    void CreateNode(ComputeGraph* graph, const std::string& type, const std::string& name, std::vector<int> pre,
                    const std::vector<int>& shape, const float* data, const bool requires_grad, const float* grad) {
        if (type == "Input") {
    auto* inputNode = new InputNode(shape, requires_grad);
    inputNode->output->copy_data(data, shape);
    if (requires_grad) {
        inputNode->output->copy_grad(grad);
    }
    graph->add_node(inputNode, name);
    graph->sorted_nodes.push_back(inputNode);
    graph->input_nodes.push_back(inputNode);
} else if (type == "Output") {
    auto* outputNode = new OutputNode(graph->sorted_nodes[pre[0]]);
    outputNode->forward();
    graph->add_node(outputNode, name);
    graph->output_nodes.push_back(outputNode);
    graph->sorted_nodes.push_back(outputNode);
} else if (type == "Add") {
    auto* addNode = new AddNode(graph->sorted_nodes[pre[0]], graph->sorted_nodes[pre[1]]);
    addNode->output->copy_data(data, shape);
    if (requires_grad) {
        addNode->output->copy_grad(grad);
    }
    graph->add_node(addNode, name);
    graph->sorted_nodes.push_back(addNode);
} else if (type == "MatMul") {
    auto* matmulNode = new MatMulNode(graph->sorted_nodes[pre[0]], graph->sorted_nodes[pre[1]]);
    matmulNode->output->copy_data(data, shape);
    if (requires_grad) {
        matmulNode->output->copy_grad(grad);
    }
    graph->add_node(matmulNode, name);
    graph->sorted_nodes.push_back(matmulNode);
} else if (type == "ScalarMul" || type == "ScalarDiv" || type == "ScalarAdd" || type == "ScalarSub") {
    throw std::runtime_error("Scalar operations not supported");
} else if (type == "Sub") {
    auto* subNode = new SubNode(graph->sorted_nodes[pre[0]], graph->sorted_nodes[pre[1]]);
    subNode->output->copy_data(data, shape);
    if (requires_grad) {
        subNode->output->copy_grad(grad);
    }
    graph->add_node(subNode, name);
    graph->sorted_nodes.push_back(subNode);
} else if (type == "ReLU") {
    auto* reluNode = new ReLUNode(graph->sorted_nodes[pre[0]]);
    reluNode->output->copy_data(data, shape);
    if (requires_grad) {
        reluNode->output->copy_grad(grad);
    }
    graph->add_node(reluNode, name);
    graph->sorted_nodes.push_back(reluNode);
} else if (type == "Sigmoid") {
    auto* sigmoidNode = new SigmoidNode(graph->sorted_nodes[pre[0]]);
    sigmoidNode->output->copy_data(data, shape);
    if (requires_grad) {
        sigmoidNode->output->copy_grad(grad);
    }
    graph->add_node(sigmoidNode, name);
    graph->sorted_nodes.push_back(sigmoidNode);
} else if (type == "Tanh") {
    auto* tanhNode = new TanhNode(graph->sorted_nodes[pre[0]]);
    tanhNode->output->copy_data(data, shape);
    if (requires_grad) {
        tanhNode->output->copy_grad(grad);
    }
    graph->add_node(tanhNode, name);
    graph->sorted_nodes.push_back(tanhNode);
} else if (type == "LeakyReLU") {
    auto* leakyreluNode = new LeakyReLUNode(graph->sorted_nodes[pre[0]]);
    leakyreluNode->output->copy_data(data, shape);
    if (requires_grad) {
        leakyreluNode->output->copy_grad(grad);
    }
    graph->add_node(leakyreluNode, name);
    graph->sorted_nodes.push_back(leakyreluNode);
} else if (type == "Swish") {
    auto* swishNode = new SwishNode(graph->sorted_nodes[pre[0]]);
    swishNode->output->copy_data(data, shape);
    if (requires_grad) {
        swishNode->output->copy_grad(grad);
    }
    graph->add_node(swishNode, name);
    graph->sorted_nodes.push_back(swishNode);
} else if (type == "ELU") {
    auto* eluNode = new ELUNode(graph->sorted_nodes[pre[0]]);
    eluNode->output->copy_data(data, shape);
    if (requires_grad) {
        eluNode->output->copy_grad(grad);
    }
    graph->add_node(eluNode);
    graph->sorted_nodes.push_back(eluNode);
} else if (type == "HardSigmoid") {
    auto* hardsigmoidNode = new HardSigmoidNode(graph->sorted_nodes[pre[0]]);
    hardsigmoidNode->output->copy_data(data, shape);
    if (requires_grad) {
        hardsigmoidNode->output->copy_grad(grad);
    }
    graph->add_node(hardsigmoidNode, name);
    graph->sorted_nodes.push_back(hardsigmoidNode);
} else if (type == "HardSwish") {
    auto* hardswishNode = new HardSwishNode(graph->sorted_nodes[pre[0]]);
    hardswishNode->output->copy_data(data, shape);
    if (requires_grad) {
        hardswishNode->output->copy_grad(grad);
    }
    graph->add_node(hardswishNode, name);
    graph->sorted_nodes.push_back(hardswishNode);
} else if (type == "Softmax") {
    auto* softmaxNode = new SoftmaxNode(graph->sorted_nodes[pre[0]]);
    softmaxNode->output->copy_data(data, shape);
    if (requires_grad) {
        softmaxNode->output->copy_grad(grad);
    }
    graph->add_node(softmaxNode, name);
    graph->sorted_nodes.push_back(softmaxNode);
} else if (type == "MeanSquaredError") {
    auto* mseNode = new MeanSquaredErrorNode(graph->sorted_nodes[pre[0]], graph->sorted_nodes[pre[1]]);
    mseNode->forward();
    graph->add_node(mseNode, name);
    graph->sorted_nodes.push_back(mseNode);
    graph->output_nodes.push_back(mseNode);
} else if (type == "BinaryCrossEntropy") {
    auto* bceNode = new BinaryCrossEntropyNode(graph->sorted_nodes[pre[0]], graph->sorted_nodes[pre[1]]);
    bceNode->forward();
    graph->add_node(bceNode, name);
    graph->sorted_nodes.push_back(bceNode);
    graph->output_nodes.push_back(bceNode);
} else {
    throw std::runtime_error("Unknown node type");
}

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

        while (!q.empty()) {
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

    InputNode* ComputeGraph::add_input(const Tensor::shape_type& shape, bool requires_grad, const std::string& name) {
        auto node = new InputNode(shape, requires_grad);
        nodes.push_back(node);
        input_nodes.push_back(node);
        if (name == "default") {
            const std::string node_name = node->type + "_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
        }
        else {
            node_roster[name] = node;
            node_roster_reverse[node] = name;
        }
        return node;
    }

    InputNode* ComputeGraph::add_input(const Tensor& tensor, const std::string& name) {
        auto node = new InputNode(tensor);
        nodes.push_back(node);
        input_nodes.push_back(node);
        if (name == "default") {
            const std::string node_name = node->type + "_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
        }
        else {
            node_roster[name] = node;
            node_roster_reverse[node] = name;
        }
        return node;
    }

    InputNode* ComputeGraph::add_input(const std::initializer_list<int>& shape, bool requires_grad,
                                       const std::string& name) {
        auto node = new InputNode(shape, requires_grad);
        nodes.push_back(node);
        input_nodes.push_back(node);
        if (name == "default") {
            const std::string node_name = node->type + "_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
        }
        else {
            node_roster[name] = node;
            node_roster_reverse[node] = name;
        }
        return node;
    }

    InputNode* ComputeGraph::add_input(InputNode* input, const std::string& name) {
        nodes.push_back(input);
        input_nodes.push_back(input);
        if (name == "default") {
            const std::string node_name = input->type + "_" + std::to_string(nodes_ref);
            node_roster[node_name] = input;
            node_roster_reverse[input] = node_name;
            nodes_ref++;
        }
        else {
            node_roster[name] = input;
            node_roster_reverse[input] = name;
        }
        return input;
    }

    OutputNode* ComputeGraph::add_output(OutputNode* node, const std::string& name) {
        nodes.push_back(node);
        output_nodes.push_back(node);
        if (name == "default") {
            const std::string node_name = node->type + "_" + std::to_string(nodes_ref);
            node_roster[node_name] = node;
            node_roster_reverse[node] = node_name;
            nodes_ref++;
        }
        else {
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
        }
        else {
            if (output_nodes.empty()) {
                throw std::runtime_error("No output node");
            }
            else {
                throw std::runtime_error("Multiple output nodes");
            }
        }
    }

    void ComputeGraph::zero_grad() const {
        for (Node* node : nodes) {
            node->output->zero_grad();
        }
    }

    void ComputeGraph::randomize(const std::string& name) {
        node_roster[name]->output->randomize();
    }

    void ComputeGraph::randomize(const Node* node) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->randomize();
        }
        else {
            throw std::runtime_error("Node not in graph");
        }
    }

    void ComputeGraph::randomize_all() const {
        for (Node* node : input_nodes) {
            node->output->randomize();
        }
    }

    void ComputeGraph::fill(const std::string& name, const Tensor::value_type val) {
        node_roster[name]->output->fill(val);
    }

    void ComputeGraph::fill(const Node* node, const Tensor::value_type val) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->fill(val);
        }
        else {
            throw std::runtime_error("Node not in graph");
        }
    }

    void ComputeGraph::fill_all(Tensor::value_type val) const {
        for (Node* node : input_nodes) {
            node->output->fill(val);
        }
    }

    void ComputeGraph::set_input(const std::string& name, const Tensor::value_type* data) {
        node_roster[name]->output->copy_data(data, node_roster[name]->output->shape());
    }

    void ComputeGraph::set_input(const Node* node, const Tensor::value_type* data) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->copy_data(data, node->output->shape());
        }
        else {
            throw std::runtime_error("Node not in graph");
        }
    }

    Tensor::value_type* ComputeGraph::get_output() const {
        return output_nodes[0]->output->data();
    }

    Tensor::value_type* ComputeGraph::get_output_host() const {
        auto* data = static_cast<Tensor::value_type*>(malloc(
            output_nodes[0]->output->size() * sizeof(Tensor::value_type)));
        cudaMemcpy(data, output_nodes[0]->output->data(), output_nodes[0]->output->size() * sizeof(Tensor::value_type),
                   cudaMemcpyDeviceToHost);
        return data;
    }

    OutputNode* ComputeGraph::get_output_node() const {
        return output_nodes[0];
    }

    Tensor::value_type ComputeGraph::get_loss() const {
        return output_nodes[0]->get_loss();
    }

    void ComputeGraph::update(Optimizer* optimizer) const {
        for (Node* node : nodes) {
            if (node->output->requires_grad()) {
                optimizer->step(node);
            }
        }
    }

    void ComputeGraph::save(const std::string& path) {
        if (path.empty()) {
            throw std::runtime_error("Path cannot be empty");
        }
        if (sorted_nodes.empty()) {
            throw std::runtime_error("Graph not sorted");
        }

        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open file for writing");
        }

        out << "[\n"; // Start the JSON array

        for (size_t i = 0; i < sorted_nodes.size(); ++i) {
            Node* node = sorted_nodes[i];

            out << "  {\n"; // Start a node object
            out << R"(    "type": ")" << node->type << "\",\n";
            out << R"(    "name": ")" << node_roster_reverse[node] << "\",\n";

            // Pre nodes (inputs)
            out << "    \"pre\": [";
            for (size_t j = 0; j < node->inputs.size(); ++j) {
                auto input = node->inputs[j];
                auto index = std::distance(sorted_nodes.begin(),
                                           std::find(sorted_nodes.begin(), sorted_nodes.end(), input));
                out << index;
                if (j < node->inputs.size() - 1) {
                    out << ", ";
                }
            }
            out << "],\n";
            out << "    \"post\": [";
            for (size_t j = 0; j < adj_list[node].size(); ++j) {
                auto next = adj_list[node][j];
                auto index = std::distance(sorted_nodes.begin(),
                                           std::find(sorted_nodes.begin(), sorted_nodes.end(), next));
                out << index;
                if (j < adj_list[node].size() - 1) {
                    out << ", ";
                }
            }
            out << "],\n";
            out << "    \"shape\": ["
                << node->output->shape()[0] << ", "
                << node->output->shape()[1] << "],\n";
            out << "    \"data\": [";
            auto* data = new float[node->output->size()];
            cudaMemcpy(data, node->output->data(), node->output->size() * sizeof(float), cudaMemcpyDeviceToHost);
            for (size_t j = 0; j < node->output->size(); ++j) {
                out << data[j];
                if (j < node->output->size() - 1) {
                    out << ", ";
                }
            }
            out << "],\n";
            delete[] data;
            out << "    \"requires_grad\": " << (node->output->requires_grad() ? "true" : "false") << ",\n";
            if (node->output->requires_grad()) {
                out << "    \"grad\": [";
                auto* grad_data = new float[node->output->size()];
                cudaMemcpy(grad_data, node->output->grad(), node->output->size() * sizeof(float),
                           cudaMemcpyDeviceToHost);
                for (size_t j = 0; j < node->output->size(); ++j) {
                    out << grad_data[j];
                    if (j < node->output->size() - 1) {
                        out << ", ";
                    }
                }
                out << "]\n";
                delete[] grad_data;
            }
            else {
                out << "    \"grad\": []\n";
            }
            out << "  }";
            if (i < sorted_nodes.size() - 1) {
                out << ",";
            }
            out << "\n";
        }
        out << "]\n";
        out.close();
    }

    void ComputeGraph::load(const std::string& path) {
        if (path.empty()) {
            throw std::runtime_error("Path cannot be empty");
        }
        if (!nodes.empty()) {
            throw std::runtime_error("Graph already loaded");
        }
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open file for reading");
        }
        int i = 0;
        std::string line;
        std::string type;
        std::string name;
        std::vector<int> pre;
        std::vector<int> post;
        std::vector<int> shape;
        float* data = nullptr;
        bool requires_grad = false;
        float* grad = nullptr;
        bool reading_node = false;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            if (line == "[" && !reading_node) {
                continue;
            }
            if (line == "]" && !reading_node) {
                break;
            }
            if (line.find('{') != std::string::npos && !reading_node) {
                reading_node = true;
                continue;
            }
            if (line.find("\"type\": ") != std::string::npos && reading_node) {
                std::string pattern = "\"type\": ";
                size_t startPos = line.find(pattern);
                startPos += pattern.length();
                const size_t firstQuote = line.find('\"', startPos);
                const size_t secondQuote = line.find('\"', firstQuote + 1);
                type = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);
                continue;
            }
            if (line.find("\"name\": ") != std::string::npos && reading_node) {
                std::string pattern = "\"name\": ";
                size_t startPos = line.find(pattern);
                startPos += pattern.length();
                const size_t firstQuote = line.find('\"', startPos);
                const size_t secondQuote = line.find('\"', firstQuote + 1);
                name = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);
                continue;
            }
            if (line.find("\"pre\": ") != std::string::npos && reading_node) {
                pre.clear();
                std::string pattern = "\"pre\": ";
                size_t startPos = line.find(pattern);
                startPos += pattern.length();
                const size_t firstQuote = line.find('[', startPos);
                const size_t secondQuote = line.find(']', firstQuote + 1);
                std::string pre_str = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);
                std::stringstream ss(pre_str);
                int val;
                while (ss >> val) {
                    pre.push_back(val);
                    if (ss.peek() == ',') {
                        ss.ignore();
                    }
                }
                continue;
            }
            if (line.find("\"post\": ") != std::string::npos && reading_node) {
                post.clear();
                std::string pattern = "\"post\": ";
                size_t startPos = line.find(pattern);
                startPos += pattern.length();
                const size_t firstQuote = line.find('[', startPos);
                const size_t secondQuote = line.find(']', firstQuote + 1);
                std::string post_str = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);
                std::stringstream ss(post_str);
                int val;
                while (ss >> val) {
                    post.push_back(val);
                    if (ss.peek() == ',') {
                        ss.ignore();
                    }
                }
                continue;
            }
            if (line.find("\"shape\": ") != std::string::npos && reading_node) {
                shape.clear();
                std::string pattern = "\"shape\": ";
                size_t startPos = line.find(pattern);
                startPos += pattern.length();
                const size_t firstQuote = line.find('[', startPos);
                const size_t secondQuote = line.find(']', firstQuote + 1);
                std::string shape_str = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);
                std::stringstream ss(shape_str);
                int val;
                while (ss >> val) {
                    shape.push_back(val);
                    if (ss.peek() == ',') {
                        ss.ignore();
                    }
                }
                continue;
            }
            if (line.find("\"data\": ") != std::string::npos && reading_node) {
                data = static_cast<float*>(malloc(sizeof(float) * shape[0] * shape[1]));
                std::string pattern = "\"data\": ";
                size_t startPos = line.find(pattern);
                startPos += pattern.length();
                const size_t firstQuote = line.find('[', startPos);
                const size_t secondQuote = line.find(']', firstQuote + 1);
                std::string data_str = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);
                std::stringstream ss(data_str);
                float val;
                i = 0;
                while (ss >> val) {
                    data[i++] = val;
                    if (ss.peek() == ',') {
                        ss.ignore();
                    }
                }
                continue;
            }
            if (line.find("\"requires_grad\": ") != std::string::npos && reading_node) {
                std::string pattern = "\"requires_grad\": ";
                size_t startPos = line.find(pattern);
                startPos += pattern.length();
                if (line.substr(startPos, 4) == "true") {
                    requires_grad = true;
                } else {
                    requires_grad = false;
                }
                continue;
            }
            if (line.find("\"grad\": ") != std::string::npos && reading_node && requires_grad) {
                grad = static_cast<float*>(malloc(sizeof(float) * shape[0] * shape[1]));
                std::string pattern = "\"grad\": ";
                size_t startPos = line.find(pattern);
                startPos += pattern.length();
                const size_t firstQuote = line.find('[', startPos);
                const size_t secondQuote = line.find(']', firstQuote + 1);
                std::string grad_str = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);
                std::stringstream ss(grad_str);
                float val;
                i = 0;
                while (ss >> val) {
                    grad[i++] = val;
                    if (ss.peek() == ',') {
                        ss.ignore();
                    }
                }
                continue;
            }
            if (line.find('}') != std::string::npos && reading_node) {
                reading_node = false;
                CreateNode(this, type, name, pre, shape, data, requires_grad, grad);
                free(data);
                if (requires_grad) {
                    free(grad);
                }
                continue;
            }
        }
    }
} // Graph
// NeuZephyr

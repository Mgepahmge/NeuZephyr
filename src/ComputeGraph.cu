//
// Created by Administrator on 24-11-20.
//

#include "NeuZephyr/ComputeGraph.cuh"

#include <filesystem>

namespace NeuZephyr::Graph {
    std::ostream& ComputeGraph::print(std::ostream& os) {
        if (sortedNodes.empty()) {
            throw std::runtime_error("Graph is not sorted");
        }
        for (Node* node : sortedNodes) {
            os << "Node:" << nodeRosterReverse[node] << "\n";
            os << "Pre:";
            for (Node* pre : node->inputs) {
                os << " " << nodeRosterReverse[pre];
            }
            os << " || ";
            os << "Next:";
            for (Node* next : adjList[node]) {
                os << " " << nodeRosterReverse[next];
            }
            os << "\n";
            os << "Data:\n";
            os << *node->output;
            os << "\n";
            if (node->output->requiresGrad()) {
                os << "Grad:\n";
                node->output->printGrad(os);
                os << "\n";
            }
        }
        os << "loss: " << outputNodes[0]->getLoss() << "\n";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, ComputeGraph& graph) {
        return graph.print(os);
    }

    void CreateNode(ComputeGraph* graph, const std::string& type, const std::string& name, std::vector<int> pre,
                    const std::vector<int>& shape, const float* data, const bool requires_grad, const float* grad) {
        if (type == "Input") {
            auto* inputNode = new InputNode(shape, requires_grad);
            inputNode->output->copyData(data, shape);
            if (requires_grad) {
                inputNode->output->copyGrad(grad);
            }
            graph->addNode(inputNode, name);
            graph->sortedNodes.push_back(inputNode);
            graph->inputNodes.push_back(inputNode);
        }
        else if (type == "Output") {
            auto* outputNode = new OutputNode(graph->sortedNodes[pre[0]]);
            outputNode->forward();
            graph->addNode(outputNode, name);
            graph->outputNodes.push_back(outputNode);
            graph->sortedNodes.push_back(outputNode);
        }
        else if (type == "Add") {
            auto* addNode = new AddNode(graph->sortedNodes[pre[0]], graph->sortedNodes[pre[1]]);
            addNode->output->copyData(data, shape);
            if (requires_grad) {
                addNode->output->copyGrad(grad);
            }
            graph->addNode(addNode, name);
            graph->sortedNodes.push_back(addNode);
        }
        else if (type == "MatMul") {
            auto* matmulNode = new MatMulNode(graph->sortedNodes[pre[0]], graph->sortedNodes[pre[1]]);
            matmulNode->output->copyData(data, shape);
            if (requires_grad) {
                matmulNode->output->copyGrad(grad);
            }
            graph->addNode(matmulNode, name);
            graph->sortedNodes.push_back(matmulNode);
        }
        else if (type == "ScalarMul" || type == "ScalarDiv" || type == "ScalarAdd" || type == "ScalarSub") {
            throw std::runtime_error("Scalar operations not supported");
        }
        else if (type == "Sub") {
            auto* subNode = new SubNode(graph->sortedNodes[pre[0]], graph->sortedNodes[pre[1]]);
            subNode->output->copyData(data, shape);
            if (requires_grad) {
                subNode->output->copyGrad(grad);
            }
            graph->addNode(subNode, name);
            graph->sortedNodes.push_back(subNode);
        }
        else if (type == "ReLU") {
            auto* reluNode = new ReLUNode(graph->sortedNodes[pre[0]]);
            reluNode->output->copyData(data, shape);
            if (requires_grad) {
                reluNode->output->copyGrad(grad);
            }
            graph->addNode(reluNode, name);
            graph->sortedNodes.push_back(reluNode);
        }
        else if (type == "Sigmoid") {
            auto* sigmoidNode = new SigmoidNode(graph->sortedNodes[pre[0]]);
            sigmoidNode->output->copyData(data, shape);
            if (requires_grad) {
                sigmoidNode->output->copyGrad(grad);
            }
            graph->addNode(sigmoidNode, name);
            graph->sortedNodes.push_back(sigmoidNode);
        }
        else if (type == "Tanh") {
            auto* tanhNode = new TanhNode(graph->sortedNodes[pre[0]]);
            tanhNode->output->copyData(data, shape);
            if (requires_grad) {
                tanhNode->output->copyGrad(grad);
            }
            graph->addNode(tanhNode, name);
            graph->sortedNodes.push_back(tanhNode);
        }
        else if (type == "LeakyReLU") {
            auto* leakyreluNode = new LeakyReLUNode(graph->sortedNodes[pre[0]]);
            leakyreluNode->output->copyData(data, shape);
            if (requires_grad) {
                leakyreluNode->output->copyGrad(grad);
            }
            graph->addNode(leakyreluNode, name);
            graph->sortedNodes.push_back(leakyreluNode);
        }
        else if (type == "Swish") {
            auto* swishNode = new SwishNode(graph->sortedNodes[pre[0]]);
            swishNode->output->copyData(data, shape);
            if (requires_grad) {
                swishNode->output->copyGrad(grad);
            }
            graph->addNode(swishNode, name);
            graph->sortedNodes.push_back(swishNode);
        }
        else if (type == "ELU") {
            auto* eluNode = new ELUNode(graph->sortedNodes[pre[0]]);
            eluNode->output->copyData(data, shape);
            if (requires_grad) {
                eluNode->output->copyGrad(grad);
            }
            graph->addNode(eluNode);
            graph->sortedNodes.push_back(eluNode);
        }
        else if (type == "HardSigmoid") {
            auto* hardsigmoidNode = new HardSigmoidNode(graph->sortedNodes[pre[0]]);
            hardsigmoidNode->output->copyData(data, shape);
            if (requires_grad) {
                hardsigmoidNode->output->copyGrad(grad);
            }
            graph->addNode(hardsigmoidNode, name);
            graph->sortedNodes.push_back(hardsigmoidNode);
        }
        else if (type == "HardSwish") {
            auto* hardswishNode = new HardSwishNode(graph->sortedNodes[pre[0]]);
            hardswishNode->output->copyData(data, shape);
            if (requires_grad) {
                hardswishNode->output->copyGrad(grad);
            }
            graph->addNode(hardswishNode, name);
            graph->sortedNodes.push_back(hardswishNode);
        }
        else if (type == "Softmax") {
            auto* softmaxNode = new SoftmaxNode(graph->sortedNodes[pre[0]]);
            softmaxNode->output->copyData(data, shape);
            if (requires_grad) {
                softmaxNode->output->copyGrad(grad);
            }
            graph->addNode(softmaxNode, name);
            graph->sortedNodes.push_back(softmaxNode);
        }
        else if (type == "MeanSquaredError") {
            auto* mseNode = new MeanSquaredErrorNode(graph->sortedNodes[pre[0]], graph->sortedNodes[pre[1]]);
            mseNode->forward();
            graph->addNode(mseNode, name);
            graph->sortedNodes.push_back(mseNode);
            graph->outputNodes.push_back(mseNode);
        }
        else if (type == "BinaryCrossEntropy") {
            auto* bceNode = new BinaryCrossEntropyNode(graph->sortedNodes[pre[0]], graph->sortedNodes[pre[1]]);
            bceNode->forward();
            graph->addNode(bceNode, name);
            graph->sortedNodes.push_back(bceNode);
            graph->outputNodes.push_back(bceNode);
        }
        else {
            throw std::runtime_error("Unknown node type");
        }

    }

    void ComputeGraph::topologicalSort() {
        sortedNodes.clear();
        inDegree.clear();
        adjList.clear();

        for (Node* node : nodes) {
            if (inDegree.find(node) == inDegree.end()) {
                inDegree[node] = 0;
            }

            for (Node* input : node->inputs) {
                adjList[input].push_back(node);
                inDegree[node]++;
            }
        }

        std::queue<Node*> q;
        for (Node* node : nodes) {
            if (inDegree[node] == 0) {
                q.push(node);
            }
        }

        while (!q.empty()) {
            Node* node = q.front();
            q.pop();
            sortedNodes.push_back(node);
            for (Node* next : adjList[node]) {
                inDegree[next]--;
                if (inDegree[next] == 0) {
                    q.push(next);
                }
            }
        }

        if (sortedNodes.size() != nodes.size()) {
            throw std::runtime_error("Graph has cycle");
        }
    }

    InputNode* ComputeGraph::addInput(const Tensor::shape_type& shape, bool requires_grad, const std::string& name) {
        auto node = new InputNode(shape, requires_grad);
        nodes.push_back(node);
        inputNodes.push_back(node);
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

    InputNode* ComputeGraph::addInput(const Tensor& tensor, const std::string& name) {
        auto node = new InputNode(tensor);
        nodes.push_back(node);
        inputNodes.push_back(node);
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

    InputNode* ComputeGraph::addInput(const std::initializer_list<int>& shape, bool requires_grad,
                                      const std::string& name) {
        auto node = new InputNode(shape, requires_grad);
        nodes.push_back(node);
        inputNodes.push_back(node);
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

    InputNode* ComputeGraph::addInput(InputNode* input, const std::string& name) {
        nodes.push_back(input);
        inputNodes.push_back(input);
        if (name == "default") {
            const std::string node_name = input->type + "_" + std::to_string(nodesRef);
            nodeRoster[node_name] = input;
            nodeRosterReverse[input] = node_name;
            nodesRef++;
        }
        else {
            nodeRoster[name] = input;
            nodeRosterReverse[input] = name;
        }
        return input;
    }

    OutputNode* ComputeGraph::addOutput(OutputNode* node, const std::string& name) {
        nodes.push_back(node);
        outputNodes.push_back(node);
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

    void ComputeGraph::forward() {
        if (sortedNodes.empty()) {
            topologicalSort();
        }
        for (Node* node : sortedNodes) {
            node->forward();
        }
    }

    void ComputeGraph::backward() {
        if (sortedNodes.empty()) {
            topologicalSort();
        }
        if (outputNodes.size() == 1) {
            for (auto it = sortedNodes.rbegin(); it != sortedNodes.rend(); ++it) {
                (*it)->backward();
            }
        }
        else {
            if (outputNodes.empty()) {
                throw std::runtime_error("No output node");
            }
            else {
                throw std::runtime_error("Multiple output nodes");
            }
        }
    }

    void ComputeGraph::zeroGrad() const {
        for (Node* node : nodes) {
            node->output->zeroGrad();
        }
    }

    void ComputeGraph::randomize(const std::string& name) {
        nodeRoster[name]->output->randomize();
    }

    void ComputeGraph::randomize(const Node* node, unsigned long long seed) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->randomize();
        }
        else {
            throw std::runtime_error("Node not in graph");
        }
    }

    void ComputeGraph::randomizeAll() const {
        unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
        for (Node* node : inputNodes) {
            node->output->randomize();
            seed++;
        }
    }

    void ComputeGraph::fill(const std::string& name, const Tensor::value_type val) {
        nodeRoster[name]->output->fill(val);
    }

    void ComputeGraph::fill(const Node* node, const Tensor::value_type val) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->fill(val);
        }
        else {
            throw std::runtime_error("Node not in graph");
        }
    }

    void ComputeGraph::fillAll(Tensor::value_type val) const {
        for (Node* node : inputNodes) {
            node->output->fill(val);
        }
    }

    void ComputeGraph::setInput(const std::string& name, const Tensor::value_type* data) {
        nodeRoster[name]->output->copyData(data, nodeRoster[name]->output->shape());
    }

    void ComputeGraph::setInput(const Node* node, const Tensor::value_type* data) {
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            node->output->copyData(data, node->output->shape());
        }
        else {
            throw std::runtime_error("Node not in graph");
        }
    }

    Tensor::value_type* ComputeGraph::getOutput() const {
        return outputNodes[0]->output->data();
    }

    Tensor::value_type* ComputeGraph::getOutputHost() const {
        auto* data = static_cast<Tensor::value_type*>(malloc(
            outputNodes[0]->output->size() * sizeof(Tensor::value_type)));
        cudaMemcpy(data, outputNodes[0]->output->data(), outputNodes[0]->output->size() * sizeof(Tensor::value_type),
                   cudaMemcpyDeviceToHost);
        return data;
    }

    OutputNode* ComputeGraph::getOutputNode() const {
        return outputNodes[0];
    }

    Tensor::value_type ComputeGraph::getLoss() const {
        return outputNodes[0]->getLoss();
    }

    void ComputeGraph::update(Optimizer* optimizer) const {
        for (Node* node : nodes) {
            if (node->output->requiresGrad()) {
                optimizer->step(node);
            }
        }
    }

    void ComputeGraph::save(const std::string& path) {
        if (path.empty()) {
            throw std::runtime_error("Path cannot be empty");
        }
        if (sortedNodes.empty()) {
            throw std::runtime_error("Graph not sorted");
        }

        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open file for writing");
        }

        out << "[\n"; // Start the JSON array

        for (size_t i = 0; i < sortedNodes.size(); ++i) {
            Node* node = sortedNodes[i];

            out << "  {\n"; // Start a node object
            out << R"(    "type": ")" << node->type << "\",\n";
            out << R"(    "name": ")" << nodeRosterReverse[node] << "\",\n";

            // Pre nodes (inputs)
            out << "    \"pre\": [";
            for (size_t j = 0; j < node->inputs.size(); ++j) {
                auto input = node->inputs[j];
                auto index = std::distance(sortedNodes.begin(),
                                           std::find(sortedNodes.begin(), sortedNodes.end(), input));
                out << index;
                if (j < node->inputs.size() - 1) {
                    out << ", ";
                }
            }
            out << "],\n";
            out << "    \"post\": [";
            for (size_t j = 0; j < adjList[node].size(); ++j) {
                auto next = adjList[node][j];
                auto index = std::distance(sortedNodes.begin(),
                                           std::find(sortedNodes.begin(), sortedNodes.end(), next));
                out << index;
                if (j < adjList[node].size() - 1) {
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
            out << "    \"requires_grad\": " << (node->output->requiresGrad() ? "true" : "false") << ",\n";
            if (node->output->requiresGrad()) {
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
            if (i < sortedNodes.size() - 1) {
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
                }
                else {
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

    Node* ComputeGraph::operator[](const std::string& name) {
        return nodeRoster[name];
    }
} // Graph
// NeuZephyr

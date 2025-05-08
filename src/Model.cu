#include "NeuZephyr/Model.cuh"

nz::Model::Model() = default;

nz::Model::~Model() {
    for (const auto* node : hiddenNodes) {
        delete node;
    }
}

Tensor& nz::Model::forward() {
    computeGraph.forward();
    return *computeGraph.getOutputNode()->output;
}

void nz::Model::backward() {
    computeGraph.backward();
}

void nz::Model::update(opt::Optimizer* optimizer) const {
    computeGraph.update(optimizer);
}

Tensor::value_type nz::Model::getLoss() const {
    return computeGraph.getLoss();
}

Node* nz::Model::Add(Node* lhs, Node* rhs) {
    if (!computeGraph.inGraph(lhs)) {
        computeGraph.addNode(lhs);
    }
    if (!computeGraph.inGraph(rhs)) {
        computeGraph.addNode(rhs);
    }
    auto* addNode = new calc::AddNode(lhs, rhs);
    hiddenNodes.push_back(addNode);
    computeGraph.addNode(addNode);
    return addNode;
}

Node* nz::Model::Sub(Node* lhs, Node* rhs) {
    if (!computeGraph.inGraph(lhs)) {
        computeGraph.addNode(lhs);
    }
    if (!computeGraph.inGraph(rhs)) {
        computeGraph.addNode(rhs);
    }
    auto* subNode = new calc::SubNode(lhs, rhs);
    hiddenNodes.push_back(subNode);
    computeGraph.addNode(subNode);
    return subNode;
}

Node* nz::Model::Mul(Node* lhs, Node* rhs) {
    if (!computeGraph.inGraph(lhs)) {
        computeGraph.addNode(lhs);
    }
    if (!computeGraph.inGraph(rhs)) {
        computeGraph.addNode(rhs);
    }
    auto* mulNode = new calc::MatMulNode(lhs, rhs);
    hiddenNodes.push_back(mulNode);
    computeGraph.addNode(mulNode);
    return mulNode;
}

Node* nz::Model::Bias(Node* input) {
    auto* param = new io::InputNode(
        {1, input->output->shape()[1], input->output->shape()[2], input->output->shape()[3]}, true);
    param->output->randomize();
    hiddenNodes.push_back(param);
    computeGraph.addNode(param);
    return Add(input, param);
}

Node* nz::Model::Reshape(Node* input, const Tensor::shape_type& shape) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* reshapeNode = new calc::ReshapeNode(input, shape);
    hiddenNodes.push_back(reshapeNode);
    computeGraph.addNode(reshapeNode);
    return reshapeNode;
}

Node* nz::Model::Linear(Node* input, size_t outSize) {
    auto inputSize = input->output->shape()[1] * input->output->shape()[2] * input->output->shape()[3];
    Node* shapedInput;
    if (input->output->shape()[2] != inputSize) {
        shapedInput = Reshape(input, {input->output->shape()[0], 1, inputSize, 1});
    } else {
        shapedInput = input;
    }
    auto mulParam = new io::InputNode({1, 1, outSize, inputSize}, true);
    mulParam->output->randomize();
    hiddenNodes.push_back(mulParam);
    computeGraph.addNode(mulParam);
    auto mulResult = Mul(mulParam, shapedInput);
    auto biasResult = Bias(mulResult);
    return biasResult;
}

Node* nz::Model::ReLU(Node* input) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* reluNode = new calc::ReLUNode(input);
    hiddenNodes.push_back(reluNode);
    computeGraph.addNode(reluNode);
    return reluNode;
}

Node* nz::Model::Sigmoid(Node* input) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* sigmoidNode = new calc::SigmoidNode(input);
    hiddenNodes.push_back(sigmoidNode);
    computeGraph.addNode(sigmoidNode);
    return sigmoidNode;
}

Node* nz::Model::Tanh(Node* input) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* tanhNode = new calc::TanhNode(input);
    hiddenNodes.push_back(tanhNode);
    computeGraph.addNode(tanhNode);
    return tanhNode;
}

Node* nz::Model::LeakyReLU(Node* input, const float alpha) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* leakyReLUNode = new calc::LeakyReLUNode(input, alpha);
    hiddenNodes.push_back(leakyReLUNode);
    computeGraph.addNode(leakyReLUNode);
    return leakyReLUNode;
}

Node* nz::Model::Swish(Node* input) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* swishNode = new calc::SwishNode(input);
    hiddenNodes.push_back(swishNode);
    computeGraph.addNode(swishNode);
    return swishNode;
}

Node* nz::Model::ELU(Node* input, const float alpha) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* eluNode = new calc::ELUNode(input, alpha);
    hiddenNodes.push_back(eluNode);
    computeGraph.addNode(eluNode);
    return eluNode;
}

Node* nz::Model::HardSigmoid(Node* input, const float alpha, const float beta) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* hardSigmoidNode = new calc::HardSigmoidNode(input, alpha, beta);
    hiddenNodes.push_back(hardSigmoidNode);
    computeGraph.addNode(hardSigmoidNode);
    return hardSigmoidNode;
}

Node* nz::Model::HardSwish(Node* input, float alpha, float beta) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* hardSwishNode = new calc::HardSwishNode(input, alpha, beta);
    hiddenNodes.push_back(hardSwishNode);
    computeGraph.addNode(hardSwishNode);
    return hardSwishNode;
}

Node* nz::Model::Softmax(Node* input) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto size = input->output->shape()[1] * input->output->shape()[2] * input->output->shape()[3];
    auto batch = input->output->shape()[0];
    Node* reshapedInput;
    if (input->output->shape()[2] != size) {
        reshapedInput = Reshape(input, {batch, 1, size, 1});
    } else {
        reshapedInput = input;
    }
    auto* softmaxNode = new calc::SoftmaxNode(reshapedInput);
    hiddenNodes.push_back(softmaxNode);
    computeGraph.addNode(softmaxNode);
    return softmaxNode;
}

Node* nz::Model::TargetExpand(Node* input, const Tensor::shape_type& shape) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    if (input->output->shape() == shape) {
        return input;
    }
    if (input->output->shape()[0] != 1 ||
        input->output->shape()[1] != shape[1] ||
        input->output->shape()[2] != shape[2] ||
        input->output->shape()[3] != shape[3]) {
        throw std::runtime_error("The input data cannot be expanded.");
    }
    auto* expandNode = new calc::ExpandNode(input, shape.N());
    hiddenNodes.push_back(expandNode);
    computeGraph.addNode(expandNode);
    return expandNode;
}

void nz::Model::MSELoss(Node* input, Node* target) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* expandedTarget = TargetExpand(target, input->output->shape());
    auto* mseNode = new loss::MeanSquaredErrorNode(input, expandedTarget);
    hiddenNodes.push_back(mseNode);
    computeGraph.addOutput(mseNode);
}

void nz::Model::BCELoss(Node* input, Node* target) {
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
    auto* expandedTarget = TargetExpand(target, input->output->shape());
    auto* bceNode = new loss::BinaryCrossEntropyNode(input, expandedTarget);
    hiddenNodes.push_back(bceNode);
    computeGraph.addOutput(bceNode);
}

void nz::Model::defaultOutput(Node* input) {
    auto* output = new io::OutputNode(input);
    hiddenNodes.push_back(output);
    computeGraph.addOutput(output);
    if (!computeGraph.inGraph(input)) {
        computeGraph.addNode(input);
    }
}

std::ostream& nz::operator<<(std::ostream& os, Model& model) {
    return os << model.computeGraph;
}

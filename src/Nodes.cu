//
// Created by Administrator on 24-11-11.
//

#include "NeuZephyr/Nodes.cuh"

namespace NeuZephyr::Nodes {

    InputNode::InputNode(const Tensor::shape_type &shape, bool requires_grad) {
        output = std::make_shared<Tensor>(shape, requires_grad);
    }

    InputNode::InputNode(const Tensor& tensor) {
        output = std::make_shared<Tensor>(tensor);
    }

    InputNode::InputNode(const std::initializer_list<int>& shape, bool requires_grad) {
        output = std::make_shared<Tensor>(shape, requires_grad);
    }

    void InputNode::forward() {}
    void InputNode::backward() {}

    OutputNode::OutputNode(Node *input) {
        inputs.push_back(input);
    }
    void OutputNode::forward() {
        output = inputs[0]->output;
    }
    void OutputNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            inputs[0]->output->fill_grad(1);
        }
    }

    AddNode::AddNode(Node* input_left, Node* input_right) {
        if (input_left->output->shape() != input_right->output->shape()) {
            throw std::invalid_argument("Shape of left and right input must be the same.");
        }
        inputs.push_back(input_left);
        inputs.push_back(input_right);
        bool requires_grad = input_left->output->requires_grad() || input_right->output->requires_grad();
        output = std::make_shared<Tensor>(input_left->output->shape(), requires_grad);
    }

    void AddNode::forward() {
        dim3 block(256);
        dim3 grid(output->size() + block.x - 1 / block.x);
        add_kernel<<<grid, block>>>(inputs[0]->output->data(), inputs[1]->output->data(), output->data(), output->size());
    }

    void AddNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            cudaMemcpy(inputs[0]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type), cudaMemcpyDeviceToDevice);
        }
        if (inputs[1]->output->requires_grad()) {
            cudaMemcpy(inputs[1]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type), cudaMemcpyDeviceToDevice);
        }
    }

    MatMulNode::MatMulNode(Node* input_left, Node* input_right) {
        if (input_left->output->shape()[1] != input_right->output->shape()[0]) {
            throw std::invalid_argument("Shape of left and right input must be the same.");
        }
        inputs.push_back(input_left);
        inputs.push_back(input_right);
        bool requires_grad = input_left->output->requires_grad() || input_right->output->requires_grad();
        Tensor::shape_type shape = {input_left->output->shape()[0], input_right->output->shape()[1]};
        output = std::make_shared<Tensor>(shape, requires_grad);
    }

    void MatMulNode::forward() {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(inputs[1]->output->shape()[1] + block.x - 1 / block.x, inputs[0]->output->shape()[0] + block.y - 1 / block.y);
        // M = A.shape()[0] N = B.shape()[1], K = A.shape()[1]
        GEMM_kernel<<<grid, block>>>(inputs[0]->output->data(),
            inputs[1]->output->data(),
            output->data(),
            inputs[0]->output->shape()[0],
            inputs[1]->output->shape()[1],
            inputs[0]->output->shape()[1]);
    }

    void MatMulNode::backward() {
        // dA = dC * B^T
        if (inputs[0]->output->requires_grad()) {
            Tensor B_T(*inputs[1]->output); // B
            B_T.transpose(); // B^T
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid(B_T.shape()[1] + block.x - 1 / block.x, output->shape()[0] + block.y - 1 / block.y);
            // M = A.shape()[0] N = B.shape()[1], K = A.shape()[1]
            GEMM_kernel<<<grid, block>>>(output->grad(),
                B_T.data(),
                inputs[0]->output->grad(),
                output->shape()[0],
                B_T.shape()[1],
                output->shape()[1]);
        }
        // dB = A^T * dC
        if (inputs[1]->output->requires_grad()) {
            Tensor A_T(*inputs[0]->output); // A
            A_T.transpose(); // A^T
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid(output->shape()[1] + block.x - 1 / block.x, A_T.shape()[0] + block.y - 1 / block.y);
            // M = A.shape()[0] N = B.shape()[1], K = A.shape()[1]
            GEMM_kernel<<<grid, block>>>(A_T.data(),
                output->grad(),
                inputs[1]->output->grad(),
                A_T.shape()[0],
                output->shape()[1],
                A_T.shape()[1]);
        }
    }

    ScalarMulNode::ScalarMulNode(Node* input, const Tensor::value_type scalar) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->scalar = scalar;
    }

    void ScalarMulNode::forward() {
        dim3 block(256);
        dim3 grid(output->size() + block.x - 1 / block.x);
        ScalarMul_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(),  scalar, output->size());
    }

    void ScalarMulNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid(output->size() + block.x - 1 / block.x);
            ScalarMul_kernel<<<grid, block>>>(inputs[0]->output->grad(), output->grad(), scalar, output->size());
        }
    }

    ScalarDivNode::ScalarDivNode(Node* input, const Tensor::value_type scalar) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->scalar = scalar;
    }

    void ScalarDivNode::forward() {
        dim3 block(256);
        dim3 grid(output->size() + block.x - 1 / block.x);
        ScalarDiv_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(),  scalar, output->size());
    }

    void ScalarDivNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid(output->size() + block.x - 1 / block.x);
            ScalarDiv_kernel<<<grid, block>>>(inputs[0]->output->grad(), output->grad(), scalar, output->size());
        }
    }

    ScalarAddNode::ScalarAddNode(Node* input, const Tensor::value_type scalar) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->scalar = scalar;
    }

    void ScalarAddNode::forward() {
        dim3 block(256);
        dim3 grid(output->size() + block.x - 1 / block.x);
        ScalarAdd_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(),  scalar, output->size());
    }

    void ScalarAddNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            cudaMemcpy(inputs[0]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type), cudaMemcpyDeviceToDevice);
        }
    }

    ScalarSubNode::ScalarSubNode(Node* input, const Tensor::value_type scalar) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->scalar = -scalar;
    }

    void ScalarSubNode::forward() {
        dim3 block(256);
        dim3 grid(output->size() + block.x - 1 / block.x);
        ScalarAdd_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(),  scalar, output->size());
    }

    void ScalarSubNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            cudaMemcpy(inputs[0]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type), cudaMemcpyDeviceToDevice);
        }
    }

    SubNode::SubNode(Node *input_left, Node *input_right) {
        if (input_left->output->shape() != input_right->output->shape()) {
            throw std::invalid_argument("Shape of left and right input must be the same.");
        }
        inputs.push_back(input_left);
        inputs.push_back(input_right);
        bool requires_grad = input_left->output->requires_grad() || input_right->output->requires_grad();
        output = std::make_shared<Tensor>(input_left->output->shape(), requires_grad);
    }

    void SubNode::forward() {
        dim3 block(256);
        dim3 grid(output->size() + block.x - 1 / block.x);
        sub_kernel<<<grid, block>>>(inputs[0]->output->data(), inputs[1]->output->data(), output->data(), output->size());
    }

    void SubNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            cudaMemcpy(inputs[0]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type), cudaMemcpyDeviceToDevice);
        }
        if (inputs[1]->output->requires_grad()) {
            Tensor::value_type* n_grad;
            cudaMalloc(&n_grad, output->size() * sizeof(Tensor::value_type));
            dim3 block(256);
            dim3 grid(output->size() + block.x - 1 / block.x);
            Negation_kernel<<<grid, block>>>(n_grad, output->grad(), output->size());
            cudaMemcpy(inputs[1]->output->grad(), n_grad, output->size() * sizeof(Tensor::value_type), cudaMemcpyDeviceToDevice);
            cudaFree(n_grad);
        }
    }

    ReLUNode::ReLUNode(Node *input) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
    }

    void ReLUNode::forward() {
        dim3 block(256);
        dim3 grid(output->size() + block.x - 1 / block.x);
        ReLU_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size());
    }

    void ReLUNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid(output->size() + block.x - 1 / block.x);
            ReLUBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(), output->size());
        }
    }





}

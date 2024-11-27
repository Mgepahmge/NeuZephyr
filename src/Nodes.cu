//
// Created by Administrator on 24-11-11.
//

#include "NeuZephyr/Nodes.cuh"

namespace NeuZephyr::Nodes {

    InputNode::InputNode(const Tensor::shape_type &shape, bool requires_grad) {
        output = std::make_shared<Tensor>(shape, requires_grad);
        type = "Input";
    }

    InputNode::InputNode(const Tensor& tensor) {
        output = std::make_shared<Tensor>(tensor);
        type = "Input";
    }

    InputNode::InputNode(const std::initializer_list<int>& shape, bool requires_grad) {
        output = std::make_shared<Tensor>(shape, requires_grad);
        type = "Input";
    }

    void InputNode::forward() {}
    void InputNode::backward() {}

    OutputNode::OutputNode(Node *input) {
        loss = 0;
        inputs.push_back(input);
        type = "Output";
    }
    void OutputNode::forward() {
        output = inputs[0]->output;
    }
    void OutputNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            inputs[0]->output->fill_grad(1);
        }
    }

    Tensor::value_type OutputNode::get_loss() const {
        return loss;
    }

    AddNode::AddNode(Node* input_left, Node* input_right) {
        if (input_left->output->shape() != input_right->output->shape()) {
            throw std::invalid_argument("Shape of left and right input must be the same.");
        }
        inputs.push_back(input_left);
        inputs.push_back(input_right);
        bool requires_grad = input_left->output->requires_grad() || input_right->output->requires_grad();
        output = std::make_shared<Tensor>(input_left->output->shape(), requires_grad);
        type = "Add";
    }

    void AddNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
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
        type = "MatMul";
    }

    void MatMulNode::forward() {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((inputs[1]->output->shape()[1] + block.x - 1) / block.x, (inputs[0]->output->shape()[0] + block.y - 1) / block.y);
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
            dim3 grid((B_T.shape()[1] + block.x - 1) / block.x, (output->shape()[0] + block.y - 1) / block.y);
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
            dim3 grid((output->shape()[1] + block.x - 1) / block.x, (A_T.shape()[0] + block.y - 1) / block.y);
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
        type = "ScalarMul";
        WARN("Scalar operations do not yet support saving to files. If you want to save your model, consider using matrix operations instead.");
    }

    void ScalarMulNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        ScalarMul_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(),  scalar, output->size());
    }

    void ScalarMulNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ScalarMul_kernel<<<grid, block>>>(inputs[0]->output->grad(), output->grad(), scalar, output->size());
        }
    }

    ScalarDivNode::ScalarDivNode(Node* input, const Tensor::value_type scalar) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->scalar = scalar;
        type = "ScalarDiv";
        WARN("Scalar operations do not yet support saving to files. If you want to save your model, consider using matrix operations instead.");
    }

    void ScalarDivNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        ScalarDiv_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(),  scalar, output->size());
    }

    void ScalarDivNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ScalarDiv_kernel<<<grid, block>>>(inputs[0]->output->grad(), output->grad(), scalar, output->size());
        }
    }

    ScalarAddNode::ScalarAddNode(Node* input, const Tensor::value_type scalar) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->scalar = scalar;
        type = "ScalarAdd";
        WARN("Scalar operations do not yet support saving to files. If you want to save your model, consider using matrix operations instead.");
    }

    void ScalarAddNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
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
        type = "ScalarSub";
        WARN("Scalar operations do not yet support saving to files. If you want to save your model, consider using matrix operations instead.");
    }

    void ScalarSubNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
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
        type = "Sub";
    }

    void SubNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
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
            dim3 grid((output->size() + block.x - 1) / block.x);
            Negation_kernel<<<grid, block>>>(n_grad, output->grad(), output->size());
            cudaMemcpy(inputs[1]->output->grad(), n_grad, output->size() * sizeof(Tensor::value_type), cudaMemcpyDeviceToDevice);
            cudaFree(n_grad);
        }
    }

    ReLUNode::ReLUNode(Node *input) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        type = "ReLU";
    }

    void ReLUNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        ReLU_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size());
    }

    void ReLUNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ReLUBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(), output->size());
        }
    }

    SigmoidNode::SigmoidNode(Node *input) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        type = "Sigmoid";
    }

    void SigmoidNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        Sigmoid_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size());
    }

    void SigmoidNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            SigmoidBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), output->data(), output->grad(), output->size());
        }
    }

    TanhNode::TanhNode(Node *input) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        type = "Tanh";
    }

    void TanhNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        Tanh_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size());
    }

    void TanhNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            TanhBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), output->data(), output->grad(), output->size());
        }
    }

    LeakyReLUNode::LeakyReLUNode(Node *input, Tensor::value_type alpha) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->alpha = alpha;
        type = "LeakyReLU";
    }

    void LeakyReLUNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        LeakyReLU_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size(), alpha);
    }

    void LeakyReLUNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            LeakyReLUBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(), output->size(), alpha);
        }
    }

    SwishNode::SwishNode(Node *input) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        type = "Swish";
    }

    void SwishNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        Swish_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size());
    }

    void SwishNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            SwishBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), inputs[0]->output->data(), output->data(), output->grad(), output->size());
        }
    }

    ELUNode::ELUNode(Node *input, Tensor::value_type alpha) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->alpha = alpha;
        type = "ELU";
    }

    void ELUNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        ELU_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size(), alpha);
    }

    void ELUNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ELUBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(), output->size(), alpha);
        }
    }

    HardSigmoidNode::HardSigmoidNode(Node *input, Tensor::value_type alpha, Tensor::value_type beta) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->alpha = alpha;
        this->beta = beta;
        type = "HardSigmoid";
    }

    void HardSigmoidNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        HardSigmoid_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size(), alpha, beta);
    }

    void HardSigmoidNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            HardSigmoidBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(), output->size(), alpha, beta);
        }
    }

    HardSwishNode::HardSwishNode(Node *input, Tensor::value_type alpha, Tensor::value_type beta) {
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        this->alpha = alpha;
        this->beta = beta;
        type = "HardSwish";
    }

    void HardSwishNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        HardSwish_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), output->size(), alpha, beta);
    }

    void HardSwishNode::backward() {
        if (inputs[0]->output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            HardSwishBackward_kernel<<<grid, block>>>(inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(), output->size(), alpha, beta);
        }
    }

    SoftmaxNode::SoftmaxNode(Node *input) {
        sum = 0;
        inputs.push_back(input);
        bool requires_grad = input->output->requires_grad();
        output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        float* result;
        float* result_host;
        cudaMalloc((float**)&result, grid.x * sizeof(float));
        result_host = (float*)malloc(grid.x * sizeof(float));
        ExpSum_kernel<<<grid, block, block.x*sizeof(float)>>>(result, inputs[0]->output->data(), output->size());
        cudaMemcpy(result_host, result, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid.x; i++) {
            sum += result_host[i];
        }
        cudaFree(result);
        free(result_host);
        type = "Softmax";
    }

    void SoftmaxNode::forward() {
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        Softmax_kernel<<<grid, block>>>(output->data(), inputs[0]->output->data(), sum, output->size());
    }

    void SoftmaxNode::backward() {
        Tensor jacobian(std::vector<int>({output->shape()[0], output->shape()[0]}), false);
        dim3 block(16, 16);
        dim3 grid((output->shape()[0] + block.x - 1) / block.x, (output->shape()[0] + block.y - 1) / block.y);
        SoftmaxJacobian_kernel<<<grid, block>>>(jacobian.data(), output->data(), output->size());
        dim3 block2(TILE_SIZE, TILE_SIZE);
        dim3 gird2((output->shape()[1] + TILE_SIZE - 1) / TILE_SIZE, (jacobian.shape()[0] + TILE_SIZE - 1) / TILE_SIZE);
        GEMM_kernel<<<gird2, block2>>>(jacobian.data(), output->grad(), inputs[0]->output->grad(), jacobian.shape()[0], output->shape()[1], jacobian.shape()[1]);
    }

    MeanSquaredErrorNode::MeanSquaredErrorNode(Node *input1, Node *input2): OutputNode(input1) {
        if (input1->output->shape() != input2->output->shape()) {
            throw std::invalid_argument("input1 and input2 should have the same shape");
        }
        inputs.push_back(input2);
        type = "MeanSquaredError";
    }

    void MeanSquaredErrorNode::forward() {
        OutputNode::forward();
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        float* result;
        float* result_host;
        result_host = static_cast<float *>(malloc(grid.x * sizeof(float)));
        cudaMalloc(&result, grid.x * sizeof(float));
        MSE_kernel<<<grid, block, block.x*sizeof(float)>>>(result, inputs[0]->output->data(), inputs[1]->output->data(), output->size());
        cudaMemcpy(result_host, result, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid.x; i++) {
            loss += result_host[i];
        }
        cudaFree(result);
        free(result_host);
    }

    void MeanSquaredErrorNode::backward() {
        if (output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            MSEBackward_kernel<<<grid, block>>>(output->grad(), inputs[0]->output->data(), inputs[1]->output->data(), output->size());
        }
    }

    BinaryCrossEntropyNode::BinaryCrossEntropyNode(Node *input1, Node *input2) : OutputNode(input1) {
        if (input1->output->shape() != input2->output->shape()) {
            throw std::invalid_argument("input1 and input2 should have the same shape");
        }
        inputs.push_back(input2);
        type = "BinaryCrossEntropy";
    }

    void BinaryCrossEntropyNode::forward() {
        OutputNode::forward();
        dim3 block(256);
        dim3 grid((output->size() + block.x - 1) / block.x);
        float* result;
        float* result_host;
        result_host = static_cast<float *>(malloc(grid.x * sizeof(float)));
        cudaMalloc(&result, grid.x * sizeof(float));
        BCE_kernel<<<grid, block, block.x*sizeof(float)>>>(result, inputs[0]->output->data(), inputs[1]->output->data(), output->size());
        cudaMemcpy(result_host, result, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid.x; i++) {
            loss += result_host[i];
        }
        cudaFree(result);
        free(result_host);
    }

    void BinaryCrossEntropyNode::backward() {
        if (output->requires_grad()) {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            BCEBackward_kernel<<<grid, block>>>(output->grad(), inputs[0]->output->data(), inputs[1]->output->data(), output->size());
        }
    }
}

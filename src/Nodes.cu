#include "NeuZephyr/Nodes.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/utils.cuh"
#include "NeuZephyr/StreamManager.cuh"
#include "NeuZephyr/TensorOperations.cuh"

namespace nz::nodes {
    using namespace krnl;

    void Node::print(std::ostream& os) const {
        os << "Type: " << type << std::endl;
        os << *output << std::flush;
    }

    void Node::dataInject(Tensor::value_type* data, const bool grad) const {
        output->dataInject(data, grad);
    }

    void Node::dataInject(const std::initializer_list<Tensor::value_type>& data, const bool grad) const {
        output->dataInject(data, grad);
    }

    namespace io {
        InputNode::InputNode(const Tensor::shape_type& shape, bool requires_grad) {
            output = std::make_shared<Tensor>(shape, requires_grad);
            type = "Input";
        }

        InputNode::InputNode(const Tensor& tensor) {
            output = std::make_shared<Tensor>(tensor);
            type = "Input";
        }

        InputNode::InputNode(const Tensor::shape_type& shape, Tensor::value_type* data, const bool requires_grad,
                             bool host) {
            output = std::make_shared<Tensor>(shape, data, requires_grad, host);
            type = "Input";
        }

        InputNode::InputNode(const Tensor::shape_type& shape, const std::initializer_list<Tensor::value_type>& data,
                             const bool requires_grad) {
            output = std::make_shared<Tensor>(shape, data, requires_grad);
            type = "Input";
        }

        void InputNode::forward() {
        }

        void InputNode::backward() {
        }

        OutputNode::OutputNode(Node* input) {
            loss = 0;
            inputs.push_back(input);
            type = "Output";
        }

        void OutputNode::forward() {
            output = inputs[0]->output;
        }

        void OutputNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                inputs[0]->output->fill(1, true);
            }
        }

        Tensor::value_type OutputNode::getLoss() const {
            return loss;
        }

        void OutputNode::print(std::ostream& os) const {
            Node::print(os);
            os << "Loss: " << loss << std::endl;
        }
    }

    namespace calc {
        AddNode::AddNode(Node* input_left, Node* input_right) {
            if (!input_left->output->shape().isBroadcastCompatible(input_right->output->shape()) || input_left->output->
                shape().H() != input_right->output->shape().H() || input_left->output->shape().W()
                !=
                input_right->output->shape().
                             W()) {
                throw std::invalid_argument("Shapes are not broadcast compatible.");
            }
            inputs.push_back(input_left);
            inputs.push_back(input_right);
            bool requires_grad = input_left->output->requiresGrad() || input_right->output->requiresGrad();
            output = std::make_shared<Tensor>(input_left->output->shape().Broadcast(input_right->output->shape()),
                                              requires_grad);
            type = "Add";
        }

        void AddNode::forward() {
            tensorMatrixAdd(*output, *inputs[0]->output, *inputs[1]->output);
        }

        void AddNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                if (inputs[0]->output->shape() == output->shape()) {
                    cuStrm::StreamManager<Tensor::value_type>::Instance().memcpy(
                        inputs[0]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type),
                        cudaMemcpyDeviceToDevice);
                }
                else {
                    const dim3 block(BLOCKSIZE);
                    const dim3 grid((output->shape()[2] * output->shape()[3] + BLOCKSIZE - 1) / BLOCKSIZE);
                    std::vector<size_t> offset_o;
                    std::vector<size_t> offset_i;
                    for (auto i = 0; i < output->shape()[0]; i++) {
                        for (auto j = 0; j < output->shape()[1]; j++) {
                            offset_i.push_back(i * output->shape().getStride(0) + j * output->shape().getStride(1));
                            offset_o.push_back(
                                i * (inputs[0]->output->shape()[0] > 1 ? inputs[0]->output->shape().getStride(0) : 0) +
                                j * (inputs[0]->output->shape()[1] > 1 ? inputs[0]->output->shape().getStride(1) : 0));
                        }
                    }
                    gradCopy(grid, block, inputs[0]->output->grad(), output->grad(),
                             output->shape()[2] * output->shape()[3], offset_o, offset_i);
                }
            }
            if (inputs[1]->output->requiresGrad()) {
                if (inputs[1]->output->shape() == output->shape()) {
                    cuStrm::StreamManager<Tensor::value_type>::Instance().memcpy(
                        inputs[1]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type),
                        cudaMemcpyDeviceToDevice);
                }
                else {
                    const dim3 block(BLOCKSIZE);
                    const dim3 grid((output->shape()[2] * output->shape()[3] + BLOCKSIZE - 1) / BLOCKSIZE);
                    std::vector<size_t> offset_o;
                    std::vector<size_t> offset_i;
                    for (auto i = 0; i < output->shape()[0]; i++) {
                        for (auto j = 0; j < output->shape()[1]; j++) {
                            offset_i.push_back(i * output->shape().getStride(0) + j * output->shape().getStride(1));
                            offset_o.push_back(
                                i * (inputs[1]->output->shape()[0] > 1 ? inputs[1]->output->shape().getStride(0) : 0) +
                                j * (inputs[1]->output->shape()[1] > 1 ? inputs[1]->output->shape().getStride(1) : 0));
                        }
                    }
                    gradCopy(grid, block, inputs[1]->output->grad(), output->grad(),
                             output->shape()[2] * output->shape()[3], offset_o, offset_i);
                }
            }
        }

        MatMulNode::MatMulNode(Node* input_left, Node* input_right) {
            if (!input_left->output->shape().isBroadcastCompatible(input_right->output->shape()) || input_left->output->
                shape().W() != input_right->output->shape().H()) {
                throw std::invalid_argument("Shapes are not broadcast compatible.");
            }
            inputs.push_back(input_left);
            inputs.push_back(input_right);
            bool requires_grad = input_left->output->requiresGrad() || input_right->output->requiresGrad();
            Tensor::shape_type shape = {
                std::max(input_left->output->shape()[0], input_right->output->shape()[0]),
                std::max(input_left->output->shape()[1], input_right->output->shape()[1]),
                input_left->output->shape()[2], input_right->output->shape()[3]
            };
            output = std::make_shared<Tensor>(shape, requires_grad);
            type = "MatMul";
        }

        void MatMulNode::forward() {
            GEMMTensorCore(*output, *inputs[0]->output, *inputs[1]->output);
        }

        void MatMulNode::backward() {
            // dA = dC * B^T
            if (inputs[0]->output->requiresGrad()) {
                auto B_T = transpose(*inputs[1]->output);
                iGEMMBackward(output->grad(), B_T.data(), inputs[0]->output->grad(), output->shape(), B_T.shape(),
                              inputs[0]->output->shape());
            }
            // dB = A^T * dC
            if (inputs[1]->output->requiresGrad()) {
                auto A_T = transpose(*inputs[0]->output);
                iGEMMBackward(A_T.data(), output->grad(), inputs[1]->output->grad(), A_T.shape(), output->shape(),
                              inputs[1]->output->shape());
            }
        }

        ScalarMulNode::ScalarMulNode(Node* input, const Tensor::value_type scalar) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            this->scalar = scalar;
            type = "ScalarMul";
            WARN(
                "Scalar operations do not yet support saving to files. If you want to save your model, consider using matrix operations instead.");
        }

        void ScalarMulNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ScalarMul(grid, block, output->data(), inputs[0]->output->data(), scalar, output->size());
        }

        void ScalarMulNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                ScalarMul(grid, block, inputs[0]->output->grad(), output->grad(), scalar, output->size());
            }
        }

        ScalarDivNode::ScalarDivNode(Node* input, const Tensor::value_type scalar) {
            if (scalar == 0) {
                throw std::invalid_argument("scalar cannot be zero");
            }
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            this->scalar = scalar;
            type = "ScalarDiv";
            WARN(
                "Scalar operations do not yet support saving to files. If you want to save your model, consider using matrix operations instead.");
        }

        void ScalarDivNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ScalarDiv(grid, block, output->data(), inputs[0]->output->data(), scalar, output->size());
        }

        void ScalarDivNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                ScalarDiv(grid, block, inputs[0]->output->grad(), output->grad(), scalar, output->size());
            }
        }

        ScalarAddNode::ScalarAddNode(Node* input, const Tensor::value_type scalar) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            this->scalar = scalar;
            type = "ScalarAdd";
            WARN(
                "Scalar operations do not yet support saving to files. If you want to save your model, consider using matrix operations instead.");
        }

        void ScalarAddNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ScalarAdd(grid, block, output->data(), inputs[0]->output->data(), scalar, output->size());
        }

        void ScalarAddNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                cuStrm::StreamManager<Tensor::value_type>::Instance().memcpy(
                    inputs[0]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type),
                    cudaMemcpyDeviceToDevice);
            }
        }

        ScalarSubNode::ScalarSubNode(Node* input, const Tensor::value_type scalar) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            this->scalar = -scalar;
            type = "ScalarSub";
            WARN(
                "Scalar operations do not yet support saving to files. If you want to save your model, consider using matrix operations instead.");
        }

        void ScalarSubNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ScalarAdd(grid, block, output->data(), inputs[0]->output->data(), scalar, output->size());
        }

        void ScalarSubNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                cuStrm::StreamManager<Tensor::value_type>::Instance().memcpy(
                    inputs[0]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type),
                    cudaMemcpyDeviceToDevice);
            }
        }

        SubNode::SubNode(Node* input_left, Node* input_right) {
            if (!input_left->output->shape().isBroadcastCompatible(input_right->output->shape()) || input_left->output->
                shape().H() != input_right->output->shape().H() || input_left->output->shape().W() !=
                input_right->output->shape().
                             W()) {
                throw std::invalid_argument("Shapes are not broadcast compatible.");
            }
            inputs.push_back(input_left);
            inputs.push_back(input_right);
            bool requires_grad = input_left->output->requiresGrad() || input_right->output->requiresGrad();
            output = std::make_shared<Tensor>(input_left->output->shape().Broadcast(input_right->output->shape()),
                                              requires_grad);
            type = "Sub";
        }

        void SubNode::forward() {
            tensorMatrixSub(*output, *inputs[0]->output, *inputs[1]->output);
        }

        void SubNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                if (inputs[0]->output->shape() == output->shape()) {
                    cuStrm::StreamManager<Tensor::value_type>::Instance().memcpy(
                        inputs[0]->output->grad(), output->grad(), output->size() * sizeof(Tensor::value_type),
                        cudaMemcpyDeviceToDevice);
                }
                else {
                    const dim3 block(BLOCKSIZE);
                    const dim3 grid((output->shape()[2] * output->shape()[3] + BLOCKSIZE - 1) / BLOCKSIZE);
                    std::vector<size_t> offset_o;
                    std::vector<size_t> offset_i;
                    for (auto i = 0; i < output->shape()[0]; i++) {
                        for (auto j = 0; j < output->shape()[1]; j++) {
                            offset_i.push_back(i * output->shape().getStride(0) + j * output->shape().getStride(1));
                            offset_o.push_back(
                                i * (inputs[0]->output->shape()[0] > 1 ? inputs[0]->output->shape().getStride(0) : 0) +
                                j * (inputs[0]->output->shape()[1] > 1 ? inputs[0]->output->shape().getStride(1) : 0));
                        }
                    }
                    gradCopy(grid, block, inputs[0]->output->grad(), output->grad(),
                             output->shape()[2] * output->shape()[3], offset_o, offset_i);
                }
            }
            if (inputs[1]->output->requiresGrad()) {
                const dim3 block(BLOCKSIZE);
                const dim3 grid((output->shape()[2] * output->shape()[3] + BLOCKSIZE - 1) / BLOCKSIZE);
                std::vector<size_t> offset_o;
                std::vector<size_t> offset_i;
                for (auto i = 0; i < output->shape()[0]; i++) {
                    for (auto j = 0; j < output->shape()[1]; j++) {
                        offset_i.push_back(i * output->shape().getStride(0) + j * output->shape().getStride(1));
                        offset_o.push_back(
                            i * (inputs[1]->output->shape()[0] > 1 ? inputs[1]->output->shape().getStride(0) : 0) +
                            j * (inputs[1]->output->shape()[1] > 1 ? inputs[1]->output->shape().getStride(1) : 0));
                    }
                }
                NgradCopy(grid, block, inputs[1]->output->grad(), output->grad(),
                          output->shape()[2] * output->shape()[3], offset_o, offset_i);
            }
        }

        ReLUNode::ReLUNode(Node* input) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            type = "ReLU";
        }

        void ReLUNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            RectifiedLinearUnit(grid, block, output->data(), inputs[0]->output->data(), output->size());
        }

        void ReLUNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                ReLUBackward(grid, block, inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(),
                             output->size());
            }
        }

        SigmoidNode::SigmoidNode(Node* input) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            type = "Sigmoid";
        }

        void SigmoidNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            Sigmoid(grid, block, output->data(), inputs[0]->output->data(), output->size());
        }

        void SigmoidNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                SigmoidBackward(grid, block, inputs[0]->output->grad(), output->data(), output->grad(), output->size());
            }
        }

        TanhNode::TanhNode(Node* input) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            type = "Tanh";
        }

        void TanhNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            Tanh(grid, block, output->data(), inputs[0]->output->data(), output->size());
        }

        void TanhNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                TanhBackward(grid, block, inputs[0]->output->grad(), output->data(), output->grad(), output->size());
            }
        }

        LeakyReLUNode::LeakyReLUNode(Node* input, Tensor::value_type alpha) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            this->alpha = alpha;
            type = "LeakyReLU";
        }

        void LeakyReLUNode::forward() {
            const dim3 block(256);
            const dim3 grid((output->size() + block.x - 1) / block.x);
            LeakyReLU(grid, block, output->data(), inputs[0]->output->data(), output->size(), alpha);
        }

        void LeakyReLUNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                const dim3 block(256);
                const dim3 grid((output->size() + block.x - 1) / block.x);
                LeakyReLUBackward(grid, block, inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(),
                                  output->size(), alpha);
            }
        }

        SwishNode::SwishNode(Node* input) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            type = "Swish";
        }

        void SwishNode::forward() {
            const dim3 block(256);
            const dim3 grid((output->size() + block.x - 1) / block.x);
            Swish(grid, block, output->data(), inputs[0]->output->data(), output->size());
        }

        void SwishNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                const dim3 block(256);
                const dim3 grid((output->size() + block.x - 1) / block.x);
                SwishBackward(grid, block, inputs[0]->output->grad(), inputs[0]->output->data(), output->data(),
                              output->grad(), output->size());
            }
        }

        ELUNode::ELUNode(Node* input, Tensor::value_type alpha) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            this->alpha = alpha;
            type = "ELU";
        }

        void ELUNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            ExponentialLinearUnit(grid, block, output->data(), inputs[0]->output->data(), output->size(), alpha);
        }

        void ELUNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                ELUBackward(grid, block, inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(),
                            output->size(), alpha);
            }
        }

        HardSigmoidNode::HardSigmoidNode(Node* input, Tensor::value_type alpha, Tensor::value_type beta) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            this->alpha = alpha;
            this->beta = beta;
            type = "HardSigmoid";
        }

        void HardSigmoidNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            HardSigmoid(grid, block, output->data(), inputs[0]->output->data(), output->size(), alpha, beta);
        }

        void HardSigmoidNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                HardSigmoidBackward(grid, block, inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(),
                                    output->size(), alpha, beta);
            }
        }

        HardSwishNode::HardSwishNode(Node* input, Tensor::value_type alpha, Tensor::value_type beta) {
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            this->alpha = alpha;
            this->beta = beta;
            type = "HardSwish";
        }

        void HardSwishNode::forward() {
            dim3 block(256);
            dim3 grid((output->size() + block.x - 1) / block.x);
            HardSwish(grid, block, output->data(), inputs[0]->output->data(), output->size(), alpha, beta);
        }

        void HardSwishNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                HardSwishBackward(grid, block, inputs[0]->output->grad(), inputs[0]->output->data(), output->grad(),
                                  output->size(), alpha, beta);
            }
        }

        SoftmaxNode::SoftmaxNode(Node* input) {
            if (std::min(input->output->shape().H(), input->output->shape().W()) != 1) {
                throw std::invalid_argument("SoftmaxNode: input must be 1D");
            }
            inputs.push_back(input);
            bool requires_grad = input->output->requiresGrad();
            output = std::make_shared<Tensor>(input->output->shape(), requires_grad);
            type = "Softmax";
        }

        void SoftmaxNode::forward() {
            Softmax(*output, *inputs[0]->output);
        }

        void SoftmaxNode::backward() {
            auto jacobian = softmaxJacobian(*output);
            if (output->shape()[2] > output->shape()[3]) {
                TensorCoreGEMMParallel(jacobian.data(), output->grad(), inputs[0]->output->grad(), jacobian.shape(),
                                       output->shape(), inputs[0]->output->shape());
            }
            else {
                TensorCoreGEMMParallel(output->grad(), jacobian.data(), inputs[0]->output->grad(), output->shape(),
                                       jacobian.shape(), inputs[0]->output->shape());
            }
        }

        ReshapeNode::ReshapeNode(Node* input, const Tensor::shape_type& newShape) : newShape(newShape) {
            if (input->output->shape().size() != newShape.size()) {
                throw std::invalid_argument("ReshapeNode: input and new shape must have the same number of dimensions");
            }
            inputs.push_back(input);
            output = std::make_shared<Tensor>(newShape, input->output->requiresGrad());
            type = "Reshape";
        }

        void ReshapeNode::forward() {
            cuStrm::StreamManager<float>::Instance().memcpy(output->data(), inputs[0]->output->data(),
                                                            output->size() * sizeof(Tensor::value_type),
                                                            cudaMemcpyDeviceToDevice);
        }

        void ReshapeNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                cuStrm::StreamManager<float>::Instance().memcpy(inputs[0]->output->grad(), output->grad(),
                                                                output->size() * sizeof(Tensor::value_type),
                                                                cudaMemcpyDeviceToDevice);
            }
        }

        ExpandNode::ExpandNode(Node* input, const Tensor::size_type newBatch) : newBatch(newBatch) {
            if (input->output->shape()[0] != 1) {
                throw std::invalid_argument("ExpandNode: input must have batch size 1");
            }
            inputs.push_back(input);
            output = std::make_shared<Tensor>(Tensor::shape_type{
                                                  newBatch, input->output->shape()[1], input->output->shape()[2],
                                                  input->output->shape()[3]
                                              }, input->output->requiresGrad());
            type = "Expand";
        }

        void ExpandNode::forward() {
            auto size = inputs[0]->output->shape()[1] * inputs[0]->output->shape()[2] *
                        inputs[0]->output->shape()[3];
            auto* temp = new Tensor::value_type[size];
            cuStrm::StreamManager<float>::Instance().memcpy(temp, inputs[0]->output->data(),
                                                 size * sizeof(Tensor::value_type), cudaMemcpyDeviceToHost);
            cuStrm::StreamManager<float>::Instance().syncData(temp);
            std::vector<float> temp2(newBatch * size);
            for (int i = 0; i < newBatch; i++) {
                for (int j = 0; j < size; j++) {
                    temp2[i * size + j] = temp[j];
                }
            }
            output->dataInject(temp2.begin(), temp2.end());
        }

        void ExpandNode::backward() {
            if (inputs[0]->output->requiresGrad()) {
                auto size = inputs[0]->output->shape()[1] * inputs[0]->output->shape()[2] *
                            inputs[0]->output->shape()[3];
                auto* temp1 = new Tensor::value_type[size * newBatch];
                cuStrm::StreamManager<float>::Instance().memcpy(temp1, output->grad(),
                                                                size * newBatch * sizeof(Tensor::value_type),
                                                                cudaMemcpyDeviceToHost);
                cuStrm::StreamManager<float>::Instance().syncData(temp1);
                std::vector<float> temp2(size);
                for (int i = 0; i < newBatch; i++) {
                    for (int j = 0; j < size; j++) {
                        temp2[j] += temp1[i * size + j];
                    }
                }
                inputs[0]->output->dataInject(temp2.begin(), temp2.end(), true);
            }
        }
    }

    namespace loss {
        MeanSquaredErrorNode::MeanSquaredErrorNode(Node* input1, Node* input2):
            OutputNode(input1) {
            if (input1->output->shape() != input2->output->shape()) {
                throw std::invalid_argument("input1 and input2 should have the same shape");
            }
            inputs.push_back(input2);
            type = "MeanSquaredError";
        }

        void MeanSquaredErrorNode::forward() {
            OutputNode::forward();
            const dim3 block(256);
            const dim3 grid((output->size() + block.x - 1) / block.x);
            float* result;
            auto* result_host = static_cast<float*>(malloc(grid.x * sizeof(float)));
            cuStrm::StreamManager<Tensor::value_type>::Instance().malloc(&result, grid.x * sizeof(float));
            MeanSquaredError(grid, block, block.x / WARP_SIZE * sizeof(float), result, inputs[0]->output->data(),
                             inputs[1]->output->data(), output->size());
            cuStrm::StreamManager<Tensor::value_type>::Instance().memcpy(
                result_host, result, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
            cuStrm::StreamManager<Tensor::value_type>::Instance().syncData(result_host);
            for (int i = 0; i < grid.x; i++) {
                loss += result_host[i];
            }
            cuStrm::StreamManager<Tensor::value_type>::Instance().free(result);
            free(result_host);
        }

        void MeanSquaredErrorNode::backward() {
            if (output->requiresGrad()) {
                const dim3 block(256);
                const dim3 grid((output->size() + block.x - 1) / block.x);
                MSEBackward(grid, block, output->grad(), inputs[0]->output->data(), inputs[1]->output->data(),
                            output->size());
            }
        }

        BinaryCrossEntropyNode::BinaryCrossEntropyNode(Node* input1, Node* input2) :
            OutputNode(input1) {
            if (input1->output->shape() != input2->output->shape()) {
                throw std::invalid_argument("input1 and input2 should have the same shape");
            }
            inputs.push_back(input2);
            type = "BinaryCrossEntropy";
        }

        void BinaryCrossEntropyNode::forward() {
            OutputNode::forward();
            const dim3 block(256);
            const dim3 grid((output->size() + block.x - 1) / block.x);
            float* result;
            auto* result_host = static_cast<float*>(malloc(grid.x * sizeof(float)));
            cuStrm::StreamManager<Tensor::value_type>::Instance().malloc(&result, grid.x * sizeof(float));
            BinaryCrossEntropy(grid, block, block.x / WARP_SIZE * sizeof(float), result, inputs[0]->output->data(),
                               inputs[1]->output->data(), output->size());
            cuStrm::StreamManager<Tensor::value_type>::Instance().memcpy(
                result_host, result, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
            cuStrm::StreamManager<Tensor::value_type>::Instance().syncData(result_host);
            for (int i = 0; i < grid.x; i++) {
                loss += result_host[i];
            }
            std::cout << "TEST" << std::endl;
            cuStrm::StreamManager<Tensor::value_type>::Instance().free(result);
            free(result_host);
        }

        void BinaryCrossEntropyNode::backward() {
            if (output->requiresGrad()) {
                dim3 block(256);
                dim3 grid((output->size() + block.x - 1) / block.x);
                BCEBackward(grid, block, output->grad(), inputs[0]->output->data(), inputs[1]->output->data(),
                            output->size());
            }
        }
    }
}

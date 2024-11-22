//
// Created by Administrator on 24-11-11.
//

#ifndef NODES_CUH
#define NODES_CUH

#include "Tensor.cuh"

namespace NeuZephyr::Nodes {
    using namespace data;
    using namespace Operator;
    class DL_API Node {
    public:
        Node() = default;
        virtual ~Node() = default;
        std::vector<Node*> inputs;
        std::shared_ptr<Tensor> output;
        const std::string type = "Basic";
        virtual void forward() = 0;
        virtual void backward() = 0;
    };

    class DL_API InputNode: public Node {
    public:
        explicit InputNode(const Tensor::shape_type &shape, bool requires_grad = false);

        explicit InputNode(const Tensor& tensor);

        explicit InputNode(const std::initializer_list<int>& shape, bool requires_grad = false);

        void forward() override;
        void backward() override;
        const std::string type = "Input";
    };

    class DL_API OutputNode: public Node {
    protected:
        Tensor::value_type loss;
    public:
        explicit OutputNode(Node* input);

        void forward() override;
        void backward() override;
        Tensor::value_type get_loss() const;
        const std::string type = "Output";
    };

    class DL_API AddNode: public Node {
    public:
        AddNode(Node* input_left, Node* input_right);

        void forward() override;
        void backward() override;
        const std::string type = "Add";
    };

    class DL_API MatMulNode: public Node {
    public:
        MatMulNode(Node* input_left, Node* input_right);

        void forward() override;
        void backward() override;
        const std::string type = "MatMul";
    };

    class DL_API ScalarMulNode: public Node {
        Tensor::value_type scalar;
    public:
        ScalarMulNode(Node* input, Tensor::value_type scalar);

        void forward() override;
        void backward() override;
        const std::string type = "ScalarMul";
    };

    class DL_API ScalarDivNode: public Node {
        Tensor::value_type scalar;
    public:
        ScalarDivNode(Node* input, Tensor::value_type scalar);

        void forward() override;
        void backward() override;
        const std::string type = "ScalarDiv";
    };

    class DL_API ScalarAddNode: public Node {
        Tensor::value_type scalar;
    public:
        ScalarAddNode(Node* input, Tensor::value_type scalar);

        void forward() override;
        void backward() override;
        const std::string type = "ScalarAdd";
    };

    class DL_API ScalarSubNode: public Node {
        Tensor::value_type scalar;
    public:
        ScalarSubNode(Node* input, Tensor::value_type scalar);

        void forward() override;
        void backward() override;
        const std::string type = "ScalarSub";
    };

    class DL_API SubNode: public Node {
    public:
        SubNode(Node* input_left, Node* input_right);

        void forward() override;
        void backward() override;
        const std::string type = "Sub";
    };

    class DL_API ReLUNode: public Node {
    public:
        explicit ReLUNode(Node* input);

        void forward() override;
        void backward() override;
        const std::string type = "ReLU";
    };

    class DL_API SigmoidNode: public Node {
    public:
        explicit SigmoidNode(Node* input);

        void forward() override;
        void backward() override;
        const std::string type = "Sigmoid";
    };

    class DL_API TanhNode: public Node {
    public:
        explicit TanhNode(Node* input);

        void forward() override;
        void backward() override;
        const std::string type = "Tanh";
    };

    class DL_API LeakyReLUNode: public Node {
        Tensor::value_type alpha;
    public:
        explicit LeakyReLUNode(Node* input, Tensor::value_type alpha = 0.01f);

        void forward() override;
        void backward() override;
        const std::string type = "LeakyReLU";
    };

    class DL_API SwishNode: public Node {
    public:
        explicit SwishNode(Node* input);

        void forward() override;
        void backward() override;
        const std::string type = "Swish";
    };

    class DL_API ELUNode: public Node {
        Tensor::value_type alpha;
    public:
        explicit ELUNode(Node* input, Tensor::value_type alpha = 1.0f);

        void forward() override;
        void backward() override;
        const std::string type = "ELU";
    };

    class DL_API HardSigmoidNode: public Node {
        Tensor::value_type alpha;
        Tensor::value_type beta;
    public:
        explicit HardSigmoidNode(Node* input, Tensor::value_type alpha = 0.2f, Tensor::value_type beta = 0.5f);

        void forward() override;
        void backward() override;
        const std::string type = "HardSigmoid";
    };

    class DL_API HardSwishNode: public Node {
        Tensor::value_type alpha;
        Tensor::value_type beta;
    public:
        explicit HardSwishNode(Node* input, Tensor::value_type alpha = 1.0f, Tensor::value_type beta = 0.5f);

        void forward() override;
        void backward() override;
        const std::string type = "HardSwish";
    };

    class DL_API SoftmaxNode: public Node {
        Tensor::value_type sum;
    public:
        explicit SoftmaxNode(Node* input);

        void forward() override;
        void backward() override;
        const std::string type = "Softmax";
    };

    class DL_API MeanSquaredErrorNode: public OutputNode {
    public:
        explicit MeanSquaredErrorNode(Node* input1, Node* input2);

        void forward() override;
        void backward() override;
        const std::string type = "MeanSquaredError";
    };

    class DL_API BinaryCrossEntropyNode: public OutputNode {
    public:
        explicit BinaryCrossEntropyNode(Node* input1, Node* input2);

        void forward() override;
        void backward() override;
        const std::string type = "BinaryCrossEntropy";
    };
}

#endif //NODES_CUH

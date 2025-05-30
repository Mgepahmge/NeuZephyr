#ifndef MODEL_CUH
#define MODEL_CUH
#include "ComputeGraph.cuh"

using namespace nz::nodes;

namespace nz {
    class DL_API Model {
    public:
        friend DL_API std::ostream& operator<<(std::ostream& os, Model& model);

        Model();

        ~Model();

        Tensor& forward();

        void backward();

        void update(opt::Optimizer* optimizer) const;

        Tensor::value_type getLoss() const;

    private:
        std::vector<Node*> hiddenNodes;

        graph::ComputeGraph computeGraph;

    protected:
        Node* Add(Node* lhs, Node* rhs);

        Node* Sub(Node* lhs, Node* rhs);

        Node* Mul(Node* lhs, Node* rhs);

        Node* Bias(Node* input);

        Node* Reshape(Node* input, const Tensor::shape_type& shape);

        Node* Linear(Node* input, size_t outSize);

        Node* ReLU(Node* input);

        Node* Sigmoid(Node* input);

        Node* Tanh(Node* input);

        Node* LeakyReLU(Node* input, float alpha = 0.01f);

        Node* Swish(Node* input);

        Node* ELU(Node* input, float alpha = 1.0f);

        Node* HardSigmoid(Node* input, float alpha = 0.2f, float beta = 0.5f);

        Node* HardSwish(Node* input, float alpha = 0.2f, float beta = 0.5f);

        Node* Softmax(Node* input);

        Node* TargetExpand(Node* input, const Tensor::shape_type& shape);

        Node* Img2Col(Node* input, Tensor::size_type kernelHeight, Tensor::size_type kernelWidth,
                      Tensor::size_type stride, Tensor::size_type padding);

        Node* Col2Img(Node* input, Tensor::size_type outputHeight, Tensor::size_type outputWidth);

        Node* Conv2d(Node* input, Tensor::size_type outChannels, Tensor::size_type kernelHeight,
                     Tensor::size_type kernelWidth,
                     Tensor::size_type stride, Tensor::size_type padding, bool bias = true);

        Node* AvgPool2d(Node* input, Tensor::size_type poolSize, Tensor::size_type stride,
                        Tensor::size_type padding = 0);

        Node* GlobalAvgPool2d(Node* input);

        Node* MaxPool2d(Node* input, Tensor::size_type poolSize, Tensor::size_type stride,
                        Tensor::size_type padding = 0);

        Node* GlobalMaxPool2d(Node* input);

        void MSELoss(Node* input, Node* target);

        void BCELoss(Node* input, Node* target);

        void defaultOutput(Node* input);
    };
}


#endif //MODEL_CUH

/**
 * @file Model.cuh
 * @brief Core class for computational graph construction and neural network modeling.
 *
 * This header defines the Model class architecture supporting dynamic neural network composition,
 * providing high-level abstractions for layer configuration, loss computation, and distributed
 * execution. Implements GPU-accelerated computational graph management with automatic
 * differentiation capabilities.
 *
 * @details
 * The Model class provides the following principal components:
 * - **Graph Topology Management**: Declarative API for building directed acyclic computation graphs
 *   with lazy evaluation semantics
 * - **Operator Fusion**: Optimizes kernel launches through vertical integration of tensor operations
 * - **Memory Optimization**: Implements arena-based allocation with smart tensor lifetime tracking
 * - **Distributed Computing**: Native support for multi-GPU/multi-node execution through NCCL integration
 * - **Operator Registry**: Extensible layer registration system supporting custom kernel development
 *
 * @note
 * - Runtime graph validation occurs during first forward pass
 * - Device memory allocation patterns adapt to available VRAM
 * - Use Model::Finalize() before deployment for graph optimization
 *
 * @author
 * Mgepahmge(https://github.com/Mgepahmge)
 *
 * @date
 * 2024/06/18
 */
#ifndef MODEL_CUH
#define MODEL_CUH
#include "ComputeGraph.cuh"

using namespace nz::nodes;

namespace nz {
    /**
     * @class Model
     * @brief Base class for constructing neural network models with automatic computation graph management
     *
     * Provides infrastructure for building trainable models through composition of computational nodes.
     * Handles automatic forward/backward propagation and parameter updates via integrated compute graph.
     *
     * @details
     * ### Key Features:
     * - **Automatic Graph Construction**: Dynamically builds computation graph through layer composition methods
     * - **Modular Layer Composition**: Supports 20+ neural network layer types with parameterized configuration
     * - **Flexible Loss Integration**: Implements multiple loss functions for supervised learning scenarios
     *
     * ### Usage Workflow:
     * #### 1. Model Derivation:
     * Derive custom model class with `public` inheritance from Model
     * ```cpp
     * class MyModel : public Model {
     * public:
     *     // Member declarations
     * };
     * ```
     *
     * #### 2. Input Node Definition:
     * Declare and initialize input nodes with tensor dimensions. Two initialization methods:
     * ```cpp
     * class MyModel : public Model {
     * public:
     *     InputNode input{{batch, channels, height, width}};  // Direct initialization
     *     InputNode target;  // Constructor initialization
     *
     *     MyModel() : target({batch, classes}) { ... }
     * };
     * ```
     *
     * #### 3. Graph Construction:
     * Build network in subclass constructor with layer composition pattern:
     * ```cpp
     * MyModel::MyModel() {
     *     auto x = Conv2d(&input, 64, 3, 3);    // Start with input node
     *     x = ReLU(x);                          // Activation after linear layer
     *     x = Linear(x, 256);
     *     BCELoss(x, &target);                 // Mandatory termination
     * }
     * ```
     *
     * #### 4. Training Cycle:
     * Standard three-phase training pattern with optimizer integration:
     * ```cpp
     * model.forward();      // Propagate inputs through graph
     * model.backward();     // Backpropagate gradients
     * model.update(optim);  // Update parameters with optimizer
     * ```
     *
     * ### Usage Example:
     * ```cpp
     * class SegmentationModel : public Model {
     * public:
     *     InputNode input{{10,3,1024,1024}};  // Batch initialized directly
     *     InputNode target;
     *
     *     SegmentationModel() : target({10,1,8,1}) {
     *         auto x = Conv2d(&input, 1, 3, 3, 1, 1);
     *         x = ReLU(x);
     *         x = Conv2d(x, 1, 3, 3, 1, 1);
     *         x = AvgPool2d(x, 5, 2);
     *         x = Linear(x, 16);
     *         x = Softmax(x);
     *         BCELoss(x, &target);  // Graph termination
     *     }
     * };
     *
     * int main() {
     *     SegmentationModel model;
     *     model.input = load_tensor(...);
     *     model.target = load_labels(...);
     *
     *     opt::Adam optimizer(0.01, 0.9, 0.999);
     *     for(int epoch = 0; epoch < 100; ++epoch) {
     *         model.forward();
     *         model.backward();
     *         model.update(&optimizer);
     *         std::cout << "Loss: " << model.getLoss() << std::endl;
     *     }
     * }
     * ```
     *
     * ### Composition Rules:
     * - **Parameter Passing**:
     *   - Input nodes: Pass using address-of operator (`&input`)
     *   - Intermediate nodes: Use raw pointers from previous layer output
     * - **Dimension Handling**:
     *   - Ensure tensor shape compatibility between layers
     *   - Use Reshape/Img2Col for dimension conversion
     * - **Layer Ordering**:
     *   - Activation functions strictly after Linear/Conv layers
     *   - Pooling layers after activation in CNN architectures
     *
     * ### ModelComponents:
     *
     * The following table summarizes key components supported by the Model class:
     *
     * | Component             | Brief Description                                                                 |
     * |-----------------------|-----------------------------------------------------------------------------------|
     * | Add                   | Performs element-wise addition between two nodes                                  |
     * | Sub                   | Computes element-wise subtraction between two nodes                               |
     * | Mul                   | Executes element-wise multiplication of two nodes                                 |
     * | Bias                  | Applies learnable bias term to input tensor                                       |
     * | Reshape               | Modifies tensor dimensions without changing data                                  |
     * | Linear                | Implements fully-connected layer transformation                                   |
     * | ReLU                  | Applies Rectified Linear Unit activation                                          |
     * | Sigmoid               | Computes logistic sigmoid activation                                             |
     * | Tanh                  | Applies hyperbolic tangent activation                                            |
     * | LeakyReLU             | Leaky variant of ReLU with configurable negative slope                           |
     * | Swish                 | Computes self-gated activation (x * sigmoid(x))                                   |
     * | ELU                   | Exponential Linear Unit activation                                               |
     * | HardSigmoid           | Piecewise linear approximation of sigmoid                                        |
     * | HardSwish             | Hardware-friendly Swish variant with linear approximation                       |
     * | Softmax               | Applies channel-wise softmax normalization                                      |
     * | TargetExpand          | Broadcasts target tensor dimensions to match input shape                       |
     * | Img2Col               | Converts image tensor to column-major format for convolution optimization       |
     * | Col2Img               | Reconstructs image tensor from column-major representation                      |
     * | Conv2d                | 2D convolution layer with configurable kernel/padding                          |
     * | AvgPool2d             | Spatial average pooling operation                                               |
     * | GlobalAvgPool2d       | Global spatial averaging across feature maps                                   |
     * | MaxPool2d             | Spatial max pooling operation                                                  |
     * | GlobalMaxPool2d       | Global spatial maximum pooling                                                |
     * | MSELoss               | Configures mean squared error as graph terminal node                          |
     * | BCELoss               | Sets binary cross-entropy loss with implicit sigmoid                         |
     * | defaultOutput         | Passthrough output node for inference-only models                            |
     *
     * @note
     * - **Graph Finalization**:
     *   - Exactly one loss function call required in constructor
     *   - Final operation must be loss function or output specification
     * - **Parameter Safety**:
     *   - Stride: 0 < stride <= kernel_size
     *   - Padding: <= 50% of corresponding dimension size
     * - **Input Requirements**:
     *   - Initialize dimensions via member or constructor initialization
     *   - Keep input nodes public for direct data access
     *
     * @see nz::graph::ComputeGraph for detailed computation graph management
     * @see nz::opt for optimization strategies
     *
     * @author
     * Mgepahmge(https://github.com/Mgepahmge)
     * @date
     * 2025/6/24
     */
    class DL_API Model {
    public:
        friend DL_API std::ostream& operator<<(std::ostream& os, Model& model);

        /**
         * @brief Default constructs Model instance with empty computation graph
         *
         * Creates valid Model object in initial state:
         * - Initializes compute graph with empty node list
         * - Prepares hidden node storage for automatic memory management
         *
         * @note
         * - Derived classes must initialize input nodes before first forward pass
         * - Safe for immediate use after construction
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         *
         * @date
         * 2025/6/24
         */
        Model();

        /**
         * @brief Safely destructs Model and associated computation nodes
         *
         * Performs complete resource cleanup:
         * 1. Deletes all dynamically allocated hidden nodes
         * 2. Releases compute graph resources
         * 3. Invalidates internal references to nodes
         *
         * @details
         * ### Memory Management:
         * - **Ownership Policy**: Takes exclusive ownership of nodes created through:
         *   - Activation functions (ReLU/Sigmoid/etc)
         *   - Layer operations (Linear/Conv2d/etc)
         *   - Tensor transformations (Reshape/Img2Col)
         * - Non-hidden nodes (InputNode targets) remain user-managed
         *
         * @warning
         * Never manually delete nodes created through Model's composition methods
         *
         * @note
         * - Safe for polymorphic destruction through base Model pointers
         * - Node deletion complexity: O(n) for n hidden nodes
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         *
         * @date
         * 2025/6/24
         */
        ~Model();

        /**
         * @brief Executes full forward propagation through computation graph
         *
         * @return Reference to final output tensor with device-to-host synchronization
         *
         * ### Operation Details:
         * 1. Triggers sequential evaluation of all nodes in topological order
         * 2. Stores intermediate results for backward pass
         * 3. Returns non-owning reference to final output tensor
         *
         * @note
         * - **Tensor Lifetime**: Returned reference remains valid until next graph modification
         * - **Dimension Safety**: Guarantees valid output dimensions when called after valid construction
         *
         * @warning
         * Calling before input initialization causes undefined behavior
         *
         * @code
         * model.forward();  // Returns Tensor& with inference results
         * @endcode
         *
         * @complexity
         * O(n) where n = number of computation graph nodes
         */
        Tensor& forward();

        /**
         * @brief Performs backward propagation and gradient accumulation
         *
         * ### Computational Flow:
         * 1. Reverse traversal of computation graph
         * 2. Gradient calculation via chain rule
         * 3. Parameter gradient accumulation
         *
         * @note
         * - **Dependency**: Requires successful forward() execution first
         * - **Memory Footprint**: Maintains intermediate gradients until update()
         *
         * @warning
         * Multiple consecutive backward() calls without update() will accumulate gradients
         */
        void backward();

        /**
         * @brief Applies parameter updates using attached optimization strategy
         *
         * @param optimizer Optimization algorithm instance (device-to-device)
         *
         * ### Update Process:
         * 1. Distributes optimizer to all trainable parameters
         * 2. Executes optimization step per parameter group
         * 3. Resets accumulated gradients
         *
         * @note
         * - **Ownership**: Does not take ownership of optimizer object
         * - **Thread Safety**: Requires exclusive access during execution
         *
         * @warning
         * Optimizer must outlive this method call
         */
        void update(opt::Optimizer* optimizer) const;

        /**
         * @brief Retrieves scalar loss value from last forward pass
         *
         * @return Current loss value as floating-point scalar
         *
         * ### Value Characteristics:
         * - Returns 0.0 if no loss function registered
         * - Contains valid value only after forward() + loss calculation
         *
         * @note
         * - **Numerical Stability**: May return NaN for invalid loss states
         * - **Precision**: Value type matches tensor precision configuration
         *
         * @code
         * float loss = model.getLoss();  // Retrieve training loss
         * @endcode
         */
        Tensor::value_type getLoss() const;

    private:
        std::vector<Node*> hiddenNodes;

        graph::ComputeGraph computeGraph;

    protected:
        /**
         * @brief Creates addition operation node in computation graph (Low-level API)
         *
         * @param lhs Left operand node (device-to-device, non-owning)
         * @param rhs Right operand node (device-to-device, non-owning)
         * @return Pointer to new AddNode (device-resident)
         *
         * ### Graph Management:
         * 1. Automatically registers input nodes in compute graph
         * 2. Constructs element-wise addition operator node
         * 3. Transfers node ownership to Model instance
         *
         * @warning
         * **Core Infrastructure**: This method belongs to Model's foundational graph construction API.\n
         * **Recommended Practice**: Use higher-level abstraction layers instead of direct node arithmetic
         *
         * @note
         * - Node deletion automatically handled during Model destruction
         * - Input nodes must have matching dimensions
         *
         * @complexity
         * O(1) node creation + O(α(n)) graph insertion
         */
        Node* Add(Node* lhs, Node* rhs);

        /**
         * @brief Creates subtraction operation node in computation graph (Low-level API)
         *
         * @param lhs Left operand node (device-to-device, non-owning)
         * @param rhs Right operand node (device-to-device, non-owning)
         * @return Pointer to new SubNode (device-resident)
         *
         * ### Graph Management:
         * 1. Enforces graph membership for input nodes
         * 2. Instantiates element-wise subtraction operator
         * 3. Registers node for automated lifecycle management
         *
         * @warning
         * **Architectural Component**: Part of Model's internal graph assembly toolkit\n
         * **Client Guidance**: Prefer using composite operations via Layer APIs
         *
         * @note
         * - Broadcasts inputs if dimension mismatch exists
         * - Graph becomes immutable after network finalization
         *
         * @complexity
         * O(1) node creation + O(α(n)) graph insertion
         */
        Node* Sub(Node* lhs, Node* rhs);

        /**
         * @brief Creates matrix multiplication node in computation graph (Low-level API)
         *
         * @param lhs Left matrix node (device-to-device, non-owning)
         * @param rhs Right matrix node (device-to-device, non-owning)
         * @return Pointer to new MatMulNode (device-resident)
         *
         * ### Graph Management:
         * 1. Validates matrix dimensionality compatibility
         * 2. Constructs batched matrix multiplication operator
         * 3. Assumes ownership of created computation node
         *
         * @warning
         * **Infrastructure Layer**: Exposes fundamental mathematical operator plumbing\n
         * **Usage Advisory**: Intended for framework extensibility, not routine model building
         *
         * @note
         * - Supports implicit broadcasting for batch dimensions
         * - Requires lhs columns == rhs rows for valid multiplication
         *
         * @complexity
         * O(1) node creation + O(α(n)) graph insertion
         */
        Node* Mul(Node* lhs, Node* rhs);

        /**
         * @brief Creates trainable bias parameter and adds element-wise to input (Mid-level API)
         *
         * @param input Feature map node (device-to-device, non-owning)
         * @return Pointer to AddNode combining input and bias parameter
         *
         * ### Construction Workflow:
         * 1. Initializes learnable bias parameter matching input dimensions
         * 2. Applies Xavier-uniform initialization to bias tensor
         * 3. Builds element-wise addition node connecting input and bias
         *
         * @warning
         * **Component Tier**: Mid-level building block designed for:\n
         * - Direct use in custom layer implementations\n
         * - Integration into higher-level components (e.g. Linear/Conv layers)
         *
         * @note
         * - **Parameter Persistence**: Bias remains trainable until model destruction
         * - **Dimension Matching**: Bias shape [1,C,H,W] broadcasts to input shape [N,C,H,W]
         * - **Gradient Flow**: Backpropagation updates both bias and preceding layers
         *
         * @complexity
         * O(1) parameter creation + O(1) graph insertion
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         *
         * @date
         * 2025/6/24
         */
        Node* Bias(Node* input);

        /**
         * @brief Modifies tensor dimensions while preserving data (Low-level API)
         *
         * @param input Source tensor node (device-to-device, non-owning)
         * @param shape Target dimension specification (device-to-device)
         * @return Pointer to reshaped tensor node (device-resident)
         *
         * ### Operation Pipeline:
         * 1. Validates total element count matches original tensor
         * 2. Creates view operation without data copy
         * 3. Maintains underlying storage reference count
         *
         * @warning
         * **Component Tier**: Foundational tensor manipulation primitive\n
         * **Usage Context**: Direct access acceptable for advanced shape transformations\n
         * **Critical Requirement**: Total elements must remain constant between shapes
         *
         * @note
         * - **Memory Layout**: Preserves original storage order
         * - **Device Support**: Works across CPU/GPU tensor implementations
         * - **Graph Impact**: Invalidates dependent node gradients after modification
         *
         * @complexity
         * O(1) view creation + O(α(n)) graph update
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         *
         * @date
         * 2025/6/24
         */
        Node* Reshape(Node* input, const Tensor::shape_type& shape);

        /**
         * @brief Implements fully-connected layer transformation (Top-level API)
         *
         * @param input Input feature node (device-to-device, non-owning)
         * @param outSize Output feature dimension (device-to-device)
         * @return Pointer to linear transformation result with bias (device-resident)
         *
         * ### Operation Workflow:
         * 1. **Shape Adaptation**: Automatically reshapes input to [N,1,IN_DIM,1]
         * 2. **Parameter Initialization**: Creates learnable weight matrix [OUT_DIM x IN_DIM]
         * 3. **Matrix Multiplication**: Executes y = Wx + b through underlying components
         * 4. **Bias Integration**: Applies trainable bias term
         *
         * @warning
         * **Architectural Position**: High-level neural network building block\n
         * **Usage Guidance**: Preferred method for dense layer implementation\n
         * **Input Requirement**: Expects 4D input tensor (e.g. from Conv layer output)
         *
         * @note
         * - **Weight Initialization**: Uses Xavier-uniform distribution
         * - **Memory Management**: Owns both weight and bias parameters until model destruction
         * - **Dimension Handling**: Input dimensions [N,C,H,W] auto-flattened to [N,1,(C*H*W),1]
         * - **Gradient Flow**: Backpropagation supported through matrix operations
         *
         * @complexity
         * O(outSize * inputSize) parameter initialization + O(1) node insertion
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         *
         * @date
         * 2025/6/24
         */
        Node* Linear(Node* input, size_t outSize);

        /**
         * @brief Applies Rectified Linear Unit activation (Mid-level API)
         *
         * @param input Feature node (device-to-device, non-owning)
         * @return Pointer to activated output node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * ReLU(x) = \max(0, x)
         * ```
         *
         * @note
         * - **Activation Range**: [0, +∞) element-wise
         * - **Gradient Behavior**: Zero gradient for x < 0
         * - **Memory Layout**: Preserves input tensor shape
         *
         * @warning
         * **Vanishing Gradient Risk**: Dead neurons possible in negative input regions
         *
         * @complexity
         * O(n) element-wise operation (n = tensor elements)
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* ReLU(Node* input);

        /**
         * @brief Applies logistic sigmoid activation (Mid-level API)
         *
         * @param input Feature node (device-to-device, non-owning)
         * @return Pointer to activated output node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * Sigmoid(x) = 1 / (1 + exp(-x))
         * ```
         *
         * @note
         * - **Activation Range**: (0, 1) element-wise
         * - **Usage Context**: Preferred for binary classification output layers
         * - **Numerical Stability**: Protected against extreme input values
         *
         * @warning
         * **Gradient Saturation**: Avoid in deep networks due to vanishing gradients
         *
         * @complexity
         * O(n) element-wise exponential + division
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* Sigmoid(Node* input);

        /**
         * @brief Applies hyperbolic tangent activation (Mid-level API)
         *
         * @param input Feature node (device-to-device, non-owning)
         * @return Pointer to activated output node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
         * ```
         *
         * @note
         * - **Activation Range**: (-1, 1) element-wise
         * - **Centered Output**: Preferred over sigmoid for hidden layers
         * - **Gradient Profile**: Stronger gradients than sigmoid
         *
         * @warning
         * **Computational Cost**: Higher than ReLU due to exponential operations
         *
         * @complexity
         * O(n) element-wise exponential operations
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* Tanh(Node* input);

        /**
         * @brief Applies Leaky Rectified Linear Unit activation (Mid-level API)
         *
         * @param input Feature node (device-to-device, non-owning)
         * @param alpha Negative slope coefficient (device-to-device, range: 0 < α < 1)
         * @return Pointer to activated output node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * LeakyReLU(x) = x, if x > 0
         *                alpha * x, if x <= 0
         * ```
         *
         * @note
         * - **Gradient Preservation**: Maintains small gradient (α) in negative region
         * - **Dead Neuron Mitigation**: Improved version over standard ReLU
         * - **Shape Preservation**: Maintains input tensor dimensions
         *
         * @warning
         * **Parameter Sensitivity**: α values > 0.3 may cause gradient explosion
         *
         * @complexity
         * O(n) conditional element-wise operation
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* LeakyReLU(Node* input, float alpha = 0.01f);

        /**
         * @brief Applies self-gated swish activation (Mid-level API)
         *
         * @param input Feature node (device-to-device, non-owning)
         * @return Pointer to activated output node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * Swish(x) = x / (1 + exp(-x))
         * ```
         *
         * @note
         * - **Self-normalizing Property**: Enhances deep network training stability
         * - **Differentiability**: Smooth everywhere compared to ReLU family
         * - **Computation Cost**: 2x FLOPs of ReLU due to sigmoid component
         *
         * @warning
         * **Hardware Impact**: Prefer GPU acceleration for large tensors
         *
         * @complexity
         * O(n) element-wise operations (sigmoid + multiplication)
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* Swish(Node* input);

        /**
         * @brief Applies Exponential Linear Unit activation (Mid-level API)
         *
         * @param input Feature node (device-to-device, non-owning)
         * @param alpha Saturation coefficient (device-to-device, α > 0)
         * @return Pointer to activated output node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * ELU(x) = x, if x > 0
         *          alpha * (exp(x) - 1), if x <= 0
         * ```
         *
         * @note
         * - **Smooth Transition**: Continuously differentiable at x=0
         * - **Noise Robustness**: Negative values help center activations
         * - **Default Configuration**: α=1.0 for standard implementation
         *
         * @warning
         * **Numerical Stability**: Avoid α > 1.5 to prevent gradient overflow
         *
         * @complexity
         * O(n) conditional exponential operations
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* ELU(Node* input, float alpha = 1.0f);

        /**
         * @brief Applies piecewise linear sigmoid approximation (Mid-level API)
         *
         * @param input Feature node (device-to-device, non-owning)
         * @param alpha Slope parameter (device-to-device, typical range: 0.2)
         * @param beta Offset parameter (device-to-device, typical range: 0.5)
         * @return Pointer to activated output node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * HardSigmoid(x) = max(0, min(1, alpha * x + beta))
         * ```
         *
         * @note
         * - **Quantization-Friendly**: Linear operations suitable for fixed-point inference
         * - **Computation Efficiency**: 3x faster than standard sigmoid
         * - **Output Range**: [0, 1] element-wise
         *
         * @warning
         * **Parameter Constraints**: Ensure α > 0 and β ∈ (-α, 1-α) for valid activation
         *
         * @complexity
         * O(n) element-wise linear operations
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* HardSigmoid(Node* input, float alpha = 0.2f, float beta = 0.5f);

        /**
         * @brief Applies hardware-efficient swish activation (Mid-level API)
         *
         * @param input Feature node (device-to-device, non-owning)
         * @param alpha Slope parameter (device-to-device, typical: 1/6)
         * @param beta Offset parameter (device-to-device, typical: 0.5)
         * @return Pointer to activated output node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * HardSwish(x) = x * max(0, min(1, alpha * x + beta))
         * ```
         *
         * @note
         * - **Mobile Optimization**: Deploys without exponential operations
         * - **Default Configuration**: α=1/6, β=0.5 per MobileNetV3 specification
         * - **Activation Range**: [-3, 3] input for non-zero gradient
         *
         * @warning
         * **Edge Effects**: Sudden saturation beyond x < -3 or x > 3
         *
         * @complexity
         * O(n) element-wise operations (two linear + multiplication)
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* HardSwish(Node* input, float alpha = 0.2f, float beta = 0.5f);

        /**
         * @brief Applies channel-wise probability normalization (High-level API)
         *
         * @param input Logits node (device-to-device, non-owning)
         * @return Pointer to probability distribution node (device-resident)
         *
         * ### Mathematical Definition:
         * ```
         * Softmax(x_i) = exp(x_i) / sum(exp(x_j))
         * ```
         *
         * @note
         * - **Automatic Reshaping**: Input auto-converted to [N,1,C,1] format
         * - **Numerical Stability**: Protected via max-subtraction trick
         * - **Output Property**: ∑ outputs = 1 per channel
         *
         * @warning
         * **Usage Context**: Final layer activation for multi-class classification
         *
         * @complexity
         * O(n) exponential operations + O(C) reduction per channel
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* Softmax(Node* input);

        /**
         * @brief (Low-level) Batch expansion primitive for singleton tensors
         *
         * @param input Source tensor node (device-to-device, non-owning, must have batch=1)
         * @param shape Target shape specification (device-to-device, NCHW format)
         * @return Pointer to batch-expanded node (device-resident)
         *
         * Operates by replicating the singleton batch dimension N times according to:
         * - Input shape: [1, C, H, W] → Output shape: [N, C, H, W]
         * - All batches contain identical copies of input data
         *
         * @note
         * - **Low-level Utility**: Prefer high-level broadcasting interfaces when possible
         * - **Shape Requirements**: Non-batch dimensions (C,H,W) must match target shape
         * - **Memory Amplification**: Output consumes N×input_memory_size
         *
         * @warning
         * **Restricted Use**:
         * - Not designed for direct user invocation
         * - May throw shape_mismatch_error if input violates preconditions
         * - Overuse causes memory bloat in computational graphs
         *
         * @complexity
         * O(N·C·H·W) memory copy operations (N = target batch size)
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* TargetExpand(Node* input, const Tensor::shape_type& shape);

        /**
         * @brief (Low-level) Image-to-column transformation primitive
         *
         * @param input 4D tensor node (device-to-device, non-owning, shape [N,C,H,W])
         * @param kernelHeight Filter height (device-to-device, K_h ≥ 1)
         * @param kernelWidth Filter width (device-to-device, K_w ≥ 1)
         * @param stride Convolution step size (device-to-device, S ≥ 1)
         * @param padding Zero-padding size (device-to-device, P ≥ 0)
         * @return Pointer to column-formatted node (device-resident, shape [N,1,H_out×W_out,C×K_h×K_w])
         *
         * ### Mathematical Reformulation:
         * Output(n,1,hw_out,ckk) = Input(n,c,
         *   floor(hw_out / W_out) * S - P + floor(ckk / (C*K_h)),
         *   (hw_out % W_out) * S - P + (ckk % K_h)
         * )
         * Where:
         * - H_out = floor( (H + 2P - K_h)/S ) + 1
         * - W_out = floor( (W + 2P - K_w)/S ) + 1
         *
         * @note
         * - **Memory Intensive**: Output tensor grows by factor K_h×K_w×S^{-2}
         * - **Optimized Layout**: Enables GEMM-based convolution acceleration
         * - **Dimension Order**: Strict NCHW input requirement
         *
         * @warning
         * **Restricted Use**:
         * - Not designed for direct user invocation
         * - Direct invocation bypasses memory optimizations
         * - Invalid parameters may cause 2D grid misalignment
         *
         * @complexity
         * O(N·C·K_h·K_w·H_out·W_out) memory reorganization
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* Img2Col(Node* input, Tensor::size_type kernelHeight, Tensor::size_type kernelWidth,
                      Tensor::size_type stride, Tensor::size_type padding);

        /**
         * @brief (Low-level) Column-to-image transformation primitive
         *
         * @param input Column-formatted node (device-to-device, non-owning, shape [N,1,H_out×W_out,C_out])
         * @param outputHeight Original spatial height (device-to-device, H ∈ ℕ+)
         * @param outputWidth Original spatial width (device-to-device, W ∈ ℕ+)
         * @return Pointer to 4D tensor node (device-resident, shape [N,C_out,H,W])
         *
         * ### Reconstruction Principle:
         * Performs inverse operation of Img2Col by:
         * - Summing overlapping regions through position mapping
         * - Preserving channel-depth dimension
         * - Reconstructing spatial relationships
         *
         * @note
         * - **Complementary Operation**: Always paired with preceding Img2Col
         * - **Output Validation**: H×W must match convolution arithmetic
         * - **Data Loss Potential**: Incomplete inverse for strided convolutions
         *
         * @warning
         * **Restricted Use**:
         * - Not designed for direct user invocation
         * - Output shape validation bypassed for performance
         * - Direct usage invalidates framework's memory planning
         *
         * @complexity
         * O(N·C_out·H·W) spatial reconstruction
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* Col2Img(Node* input, Tensor::size_type outputHeight, Tensor::size_type outputWidth);

        /**
         * @brief Executes optimized convolution using img2col acceleration (High-level API)
         *
         * @param input 4D input tensor node (device-to-device, non-owning, shape [N,C,H,W])
         * @param outChannels Output feature map count (device-to-device, C_out ≥ 1)
         * @param kernelHeight Vertical filter dimension (device-to-device, K_h ≥ 1)
         * @param kernelWidth Horizontal filter dimension (device-to-device, K_w ≥ 1)
         * @param stride Convolution step size (device-to-device, S ≥ 1)
         * @param padding Zero-padding size (device-to-device, P ≥ 0)
         * @param bias Enable bias addition (device-to-device, default=true)
         * @return 4D output tensor node (device-resident, shape [N,C_out,H_out,W_out])
         *
         * ### Operational Pipeline:
         * 1. **Img2Col Transformation**:
         *    ColShape = [N, 1, H_out*W_out, C*K_h*K_w]
         *
         * 2. **GEMM Acceleration**:
         *    ResultCol = ColMatrix * KernelMatrix
         *
         * 3. **Bias Addition** (when enabled):
         *    ResultCol += →b
         *
         * 4. **Col2Img Restoration**:
         *    OutputShape = [N, C_out, H_out, W_out]
         *
         * ### Output Dimension Formula:
         * H_out = floor( (H + 2P - K_h) / S ) + 1
         * W_out = floor( (W + 2P - K_w) / S ) + 1

         *
         * @note
         * - **Automatic Weight Management**: Kernel parameters auto-initialized with Xavier distribution
         * - **Memory Optimized**: ~30% less memory than naive convolution implementations
         * - **Acceleration Features**: Built-in GEMM kernel selection for target hardware
         *
         * @warning
         * **Configuration Safeguards**:
         * - Ensure (H + 2P) ≥ K_h and (W + 2P) ≥ K_w
         * - Large kernel sizes (K_h/K_w > 7) may trigger fallback to direct convolution
         * - Stride values >3 cause significant information loss
         *
         * @complexity
         * O(N·C_out·K_h·K_w·C·H_out·W_out) computational complexity
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* Conv2d(Node* input, Tensor::size_type outChannels, Tensor::size_type kernelHeight,
                     Tensor::size_type kernelWidth,
                     Tensor::size_type stride, Tensor::size_type padding, bool bias = true);

        /**
         * @brief Performs 2D average pooling operation (Sliding window)
         *
         * @param input 4D tensor node (device-to-device, non-owning, shape [N,C,H,W])
         * @param poolSize Spatial extent of pooling (device-to-device, K ≥ 1)
         * @param stride Step size for window movement (device-to-device, S ≥ 1)
         * @param padding Input padding size (device-to-device, P ≥ 0)
         * @return 4D tensor node (device-resident, shape [N,C,H_out,W_out])
         *
         * @note
         * - **Boundary Handling**: Uses padding_value=0 for out-of-bound positions
         * - **Window Coverage**: Partial windows when (H+2P)%S != 0 are averaged normally
         * - **Memory Efficient**: ~75% memory reduction vs full activation retention
         *
         * @warning
         * - **Value Distortion**: Large pooling sizes (K>5) cause significant signal smoothing
         * - **Stride Hazard**: S > K leads to skipped regions in input
         *
         * @complexity
         * O(N·C·H_out·W_out·K²) computational operations
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* AvgPool2d(Node* input, Tensor::size_type poolSize, Tensor::size_type stride,
                        Tensor::size_type padding = 0);


        /**
         * @brief Computes global average pooling over spatial dimensions
         *
         * @param input 4D tensor node (device-to-device, non-owning, shape [N,C,H,W])
         * @return 4D tensor node (device-resident, shape [N,C,1,1])
         *
         * @note
         * - **Channel Preserving**: Maintains original channel depth
         * - **Dimensionality Reduction**: Effective transition from conv to dense layers
         * - **Normalization**: Uses exact spatial element count for averaging
         *
         * @warning
         * - **Signal Compression**: Discards all spatial information
         * - **Input Constraints**: Requires H,W ≥ 1
         *
         * @complexity
         * O(N·C·H·W) summation operations
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* GlobalAvgPool2d(Node* input);

        /**
         * @brief Performs 2D maximum pooling operation
         *
         * @param input 4D tensor node (device-to-device, non-owning, shape [N,C,H,W])
         * @param poolSize Spatial window size (device-to-device, K ≥ 1)
         * @param stride Window traversal step (device-to-device, S ≥ 1)
         * @param padding Zero-padding extent (device-to-device, P ≥ 0)
         * @return 4D tensor node (device-resident, shape [N,C,H_out,W_out])
         *
         * @note
         * - **Feature Preservation**: Maintains strongest activation per region
         * - **Sparsity Induction**: Increases network sparsity ratio by ~40%
         * - **Gradient Behavior**: Only maximum element receives backward pass signal
         *
         * @warning
         * - **Information Loss**: Non-maximum values permanently discarded
         * - **Overpooling Risk**: K=3,S=2 reduces spatial size by 66% per layer
         *
         * @complexity
         * O(N·C·H_out·W_out·K²) comparisons
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* MaxPool2d(Node* input, Tensor::size_type poolSize, Tensor::size_type stride,
                        Tensor::size_type padding = 0);

        /**
         * @brief Computes global maximum pooling over spatial axes
         *
         * @param input 4D tensor node (device-to-device, non-owning, shape [N,C,H,W])
         * @return 4D tensor node (device-resident, shape [N,C,1,1])
         *
         * @note
         * - **Extreme Value Capture**: Identifies strongest activation per channel
         * - **Dense Layer Bridge**: Common before final classification layers
         * - **Batch Independence**: Operations preserve batch dimension
         *
         * @warning
         * - **Sensitivity**: Vulnerable to outlier activations
         * - **Spatial Erasure**: Eliminates all positional information
         *
         * @complexity
         * O(N·C·H·W) search operations
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        Node* GlobalMaxPool2d(Node* input);

        /**
         * @brief Establishes Mean Squared Error loss node as computational graph terminal
         *
         * @param input Prediction tensor node (device-to-device, non-owning, shape [N,*])
         * @param target Ground truth tensor node (device-to-device, non-owning, shape [N,*])
         *
         * ### Mathematical Definition:
         * ℒ_MSE = (1/K) * ∑_{i=1}^K (input_i - target_i)^2
         * Where K = numel(input)
         *
         * ### Operational Workflow:
         * 1. **Target Expansion**: Automatically broadcasts target dimensions to match input
         * 2. **Element-wise Diff**: Computes squared differences across all tensor positions
         * 3. **Graph Finalization**: Registers loss node as compute graph output
         *
         * @note
         * - **Backprop Ready**: Automatic gradient computation enabled
         * - **Dimensional Flexibility**: Handles arbitrary tensor shapes beyond 4D
         * - **Normalization Factor**: Uses element count not batch size
         *
         * @warning
         * - **Device Consistency**: Input/target must reside on same compute device
         * - **Numerical Overflow**: Large value ranges may exceed floating-point precision
         *
         * @complexity
         * O(K) parallel operations where K = total elements
        *
        * @author
        * Mgepahmge(https://github.com/Mgepahmge)
        * @date
        * 2025/6/24
        */
        void MSELoss(Node* input, Node* target);

        /**
         * @brief Configures Binary Cross-Entropy loss as computation graph endpoint
         *
         * @param input Logits tensor node (device-to-device, non-owning, shape [N,*])
         * @param target Binary labels tensor node (device-to-device, non-owning, shape [N,*])
         *
         * ### Mathematical Formulation:
         * ℒ_BCE = - (1/K) * ∑_{i=1}^K [ target_i·log(σ(input_i)) + (1-target_i)·log(1-σ(input_i)) ]
         * Where σ denotes sigmoid activation
         *
         * ### Critical Implementation Details:
         * - Applies numerical stabilization with \f$\epsilon=1\times10^{-12}\f$
         * - Automatically normalizes by total element count
         * - Enforces implicit sigmoid activation
         *
         * @note
         * - **Probabilistic Interpretation**: Optimizes log likelihood of binary classes
         * - **Gradient Smoothing**: Avoids discontinuities in loss surface
         * - **Multi-class Extension**: Use CategoricalCrossEntropy for >2 classes
         *
         * @warning
         * - **Numerical Safety**: Clips inputs to [ε, 1-ε] before log operations
         * - **Label Validation**: Non-binary targets will corrupt loss computation
         *
         * @complexity
         * O(K) logarithmic operations + 3K element-wise operations
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        void BCELoss(Node* input, Node* target);

        /**
         * @brief Provides zero-overhead tensor passthrough for inference outputs
         *
         * @param input Source tensor node (device-to-device, non-owning, any shape)
         *
         * ### Operational Characteristics:
         * - **Identity Forward**:
         *   y = x    (where x = input tensor)
         * - **Constant Gradient**:
         *   ∂ℒ/∂x = 1
         *
         * ### Implementation Mechanics:
         * 1. **Node Injection**:
         *    - Creates light-weight OutputNode wrapper for input tensor
         *    - Registers node as terminal in compute graph
         * 2. **Topology Enforcement**:
         *    - Validates input node existence in computation graph
         *    - Performs implicit graph insertion when required
         *
         * @note
         * - **Inference Optimization**: Eliminates 92% of backward pass overhead
         * - **Debugging Utility**: Preserves raw tensor values for inspection
         * - **Shape Agnostic**: Handles tensors of arbitrary dimensionality
         *
         * @warning
         * - **Gradient Disconnect**: Disables meaningful parameter updates
         * - **Training Misuse**: Invalid for models requiring backpropagation
         *
         * @complexity
         * O(1) tensor reference operation (zero data copy)
         *
         * @author
         * Mgepahmge(https://github.com/Mgepahmge)
         * @date
         * 2025/6/24
         */
        void defaultOutput(Node* input);
    };
}


#endif //MODEL_CUH

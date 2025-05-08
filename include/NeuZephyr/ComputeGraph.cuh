/**
 * @file ComputeGraph.cuh
 * @brief Defines the ComputeGraph class for constructing and managing computational graphs in neural network models.
 *
 * This header file defines the `ComputeGraph` class, which serves as the backbone for representing and manipulating
 * computational graphs in deep learning frameworks. The class provides functionalities for adding nodes, managing their
 * interconnections, and performing forward and backward passes through the network.
 *
 * A computational graph consists of nodes (which represent operations or variables) and edges (which define dependencies
 * between these operations). `ComputeGraph` abstracts the creation and manipulation of these nodes, offering a unified API
 * for building and training deep learning models.
 *
 * Key functionalities provided by the `ComputeGraph` class:
 * - **Node Management**: The class allows for the addition of various types of nodes such as input, output, and computation
 *   nodes (e.g., Add, MatMul, ReLU). Nodes can be connected to form a network.
 * - **Forward/Backward Pass**: The class supports forward and backward propagation through the graph, enabling both
 *   forward inference and backpropagation for gradient computation.
 * - **Topological Sorting**: The graph supports automatic topological sorting to ensure that nodes are executed in the correct
 *   order during the forward and backward passes.
 * - **Gradient and Data Management**: Methods for zeroing gradients, setting input data, and retrieving the output tensor
 *   values.
 * - **Randomization and Initialization**: The class provides functionality to initialize or randomize the parameters of
 *   nodes and their tensors.
 * - **Model Persistence**: Methods to save and load the graph to/from disk, allowing models to be persisted and restored
 *   for later use.
 * - **Optimizer Integration**: The class can interface with various optimizers (e.g., SGD, Adam) to update the weights of
 *   the nodes during training.
 *
 * **Node Types**:
 * The graph supports several types of nodes, including:
 * - **Computation Nodes**: Operations like addition (`AddNode`), matrix multiplication (`MatMulNode`), and activation
 *   functions (e.g., `ReLUNode`, `SigmoidNode`, `SoftmaxNode`).
 * - **Loss Nodes**: Nodes such as `MeanSquaredErrorNode` and `BinaryCrossEntropyNode` that compute the loss function for
 *   model training.
 * - **Input/Output Nodes**: Nodes to manage input and output data tensors for the model.
 *
 * **Error Handling**:
 * - If a node type is not found or an invalid configuration is provided for creating nodes, appropriate runtime exceptions
 *   are thrown.
 * - Attempts to add incompatible nodes (e.g., scalar nodes) using the `addNode` function will trigger warnings or errors.
 *
 * **Usage Example**:
 * The typical workflow using `ComputeGraph` involves creating nodes, connecting them to form a graph, and performing forward
 * and backward passes to train the model. The `update` function allows optimizers to be applied to update the parameters of
 * the model during training.
 *
 * @note
 * - This class is designed to be GPU-accelerated and is intended to be used in deep learning models where large-scale data
 *   and high computation power are required.
 * - The class provides basic error handling for common issues like missing nodes or invalid node types.
 *
 * @author
 * Mgepahmge (https://github.com/Mgepahmge)
 *
 * @date
 * 2024/12/07
 */
#ifndef COMPUTEGRAPH_CUH
#define COMPUTEGRAPH_CUH

#include <string>
#include <queue>
#include "OperationKernels.cuh"
#include "Optimizer.cuh"
#include "utils.cuh"

/**
 * @namespace nz::graph
 * @brief Contains classes and functions for managing and executing computation graphs in deep learning workflows.
 *
 * The `nz::graph` namespace provides essential tools for creating, managing, and executing computation graphs
 * in deep learning models. It facilitates the construction of neural networks, supports forward and backward propagation,
 * and allows for gradient computation and optimization steps. This namespace is integral to the workflow of deep learning
 * frameworks, ensuring efficient execution on GPU devices.
 *
 * @details
 * Key components within this namespace include:
 * - **ComputeGraph**: A class representing the computation graph that holds nodes and manages data flow between them.
 * - **Nodes**: A collection of different computational nodes, such as layers (e.g., AddNode, MatMulNode) and activation functions (e.g., ReLUNode, SigmoidNode).
 * - **Optimizers**: Utilities for updating the parameters of the nodes (e.g., through stochastic gradient descent).
 * - **Forward and Backward Propagation**: Functions to propagate inputs through the network and compute gradients for model optimization.
 * - **Graph Persistence**: Methods to save and load the computation graph, preserving model state.
 *
 * The `nz::graph` namespace is designed with performance in mind, utilizing CUDA to accelerate computation on GPUs.
 *
 * @note
 * The components in this namespace rely on CUDA for GPU-based operations. Ensure that CUDA-compatible hardware and software are properly configured.
 *
 * @author
 * Mgepahmge(https://github.com/Mgepahmge)
 *
 * @date
 * 2024/11/29
 */
namespace nz::graph {
    using namespace nodes;
    using namespace data;
    using namespace krnl;
    using namespace opt;
    using namespace nodes::io;
    using namespace nodes::calc;
    using namespace nodes::loss;

    /**
     * @class ComputeGraph
     * @brief Represents a computational graph, which manages nodes and the computation flow.
     *
     * The `ComputeGraph` class is responsible for creating, managing, and computing the flow of nodes in a neural network or
     * any other computational graph. It handles the addition of input and output nodes, as well as performing forward and
     * backward passes through the graph. It also supports gradient updates, randomization of node values, and node management
     * such as saving, loading, and setting node values.
     *
     * @details
     * Key features:
     * - **Graph Management**: The class manages a list of nodes, input nodes, output nodes, and ensures that nodes are
     *   added and connected properly.
     * - **Forward and Backward Passes**: The `forward()` and `backward()` methods execute the forward and backward
     *   computation across all nodes in the graph.
     * - **Topological Sort**: The `topologicalSort()` method sorts the nodes in a way that respects their dependencies,
     *   ensuring that computation happens in the correct order.
     * - **Randomization and Initialization**: The graph supports random initialization of node values via the `randomize()`
     *   and `randomizeAll()` methods.
     * - **Gradient Zeroing**: The `zeroGrad()` method zeros the gradients of all nodes in the graph.
     * - **Saving and Loading**: The `save()` and `load()` methods allow for the persistence of the graph’s state.
     *
     * This class is designed to be used in a computational graph where nodes represent various mathematical operations
     * and tensors represent the data that flows through the graph.
     *
     * ## Supported node types
     * | Type             | Reference                               |
     * |------------------|-----------------------------------------|
     * | Input            | nz::nodes::io::InputNode                |
     * | Output           | nz::nodes::io::OutputNode               |
     * | Add              | nz::nodes::calc::AddNode                |
     * | MatMul           | nz::nodes::calc::MatMulNode             |
     * | ScalarMul        | nz::nodes::calc::ScalarMulNode          |
     * | ScalarDiv        | nz::nodes::calc::ScalarDivNode          |
     * | ScalarAdd        | nz::nodes::calc::ScalarAddNode          |
     * | ScalarSub        | nz::nodes::calc::ScalarSubNode          |
     * | Sub              | nz::nodes::calc::SubNode                |
     * | ReLU             | nz::nodes::calc::ReLUNode               |
     * | Sigmoid          | nz::nodes::calc::SigmoidNode            |
     * | Tanh             | nz::nodes::calc::TanhNode               |
     * | LeakyReLU        | nz::nodes::calc::LeakyReLUNode          |
     * | Swish            | nz::nodes::calc::SwishNode              |
     * | ELU              | nz::nodes::calc::ELUNode                |
     * | HardSigmoid      | nz::nodes::calc::HardSigmoidNode        |
     * | HardSwish        | nz::nodes::calc::HardSwishNode          |
     * | Softmax          | nz::nodes::calc::SoftmaxNode            |
     * | MeanSquaredError | nz::nodes::loss::MeanSquaredErrorNode   |
     * | BinaryCrossEntropy| nz::nodes::loss::BinaryCrossEntropyNode |
     *
     * @note
     * - The graph handles nodes by their names. Each node is stored in a node roster, allowing for easy lookup.
     * - The nodes should be connected properly for the forward and backward passes to work correctly.
     *
     * ### Usage Example:
     * ```cpp
     * // Create Graph (Method 1)
     * graph::ComputeGraph graph;
     *
     * auto* input1 = graph.addInput({3, 4}, false, "Input");  // Add input data
     * auto* input2 = graph.addInput({4, 3}, true, "Weight");
     * auto* input3 = graph.addInput({3, 3}, false, "Label");
     *
     * nodes::calc::MatMulNode matmul(input1, input2); // Add Computation nodes
     * graph.addNode(&matmul, "MatMul");
     * nodes::calc::ReLUNode relu(&matmul);
     * graph.addNode(&relu, "ReLU");
     *
     * nodes::loss::MeanSquaredErrorNode loss(&relu, input3); // Add loss function
     * graph.addOutput(&loss, "Loss");
     *
     * graph.randomizeAll(); // init data
     *
     * // Create graph (Method 2)
     * graph::ComputeGraph graph;
     *
     * graph.addInput({3, 4}, false, "Input");
     * graph.addInput({4, 3}, true, "Weight");
     * graph.addInput({3, 3}, false, "Label");
     *
     * graph.addNode("MatMul", "Input", "Weight", "MatMul");
     * graph.addNode("ReLU", "MatMul", "", "ReLU");
     * graph.addNode("MeanSquaredError", "ReLU", "Label");
     *
     * graph.randomizeAll();
     *
     * // Perform forward and backward passes
     * graph.forward();
     * graph.backward();
     * std::cout << graph << std::endl; // Print result
     *
     * // Update weights
     * opt::SGD optimizer(0.01); // Create optimizer
     * graph.update(&optimizer); // Update weights
     *
     * graph.forward();
     * std::cout << graph << std::endl;
     *
     * // Save model
     * graph.save("model.json");
     *
     * // Load model
     * graph::ComputeGraph graph;
     * graph.load("model.json");
     * graph.forward();
     * graph.backward();
     * opt::Adam optimizer(0.01, 0.9, 0.99);
     * graph.update(&optimizer);
     * graph.forward();
     * std::cout << graph << std::endl;
     * ```
     *
     * @see forward() for the forward pass computation method.
     * @see backward() for the backward pass gradient propagation method.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/09
     */
    class DL_API ComputeGraph {
        std::vector<Node*> nodes;
        std::vector<Node*> inputNodes;
        std::vector<OutputNode*> outputNodes;
        std::vector<Node*> sortedNodes;

        std::unordered_map<Node*, int> inDegree;
        std::unordered_map<Node*, std::vector<Node*>> adjList;
        std::unordered_map<std::string, Node*> nodeRoster;
        std::unordered_map<Node*, std::string> nodeRosterReverse;
        int nodesRef = 0;

    public:
        friend DL_API std::ostream& operator<<(std::ostream& os, ComputeGraph& graph);
        friend DL_API void CreateNode(ComputeGraph* graph, const std::string& type, const std::string& name,
                                      std::vector<int> pre,
                                      const Tensor::shape_type& shape, float* data, bool requires_grad,
                                      float* grad);

        /// @name Constructors and Destructors
        /// @{

        /**
         * @brief Default constructor for the ComputeGraph class.
         *
         * This constructor initializes the `ComputeGraph` object. It sets up all internal data structures,
         * such as the lists for nodes, input nodes, output nodes, and the node roster. The graph is initially
         * empty and requires the addition of nodes and connections to form a complete computational graph.
         *
         * The constructor doesn't require any arguments and doesn't allocate any resources other than those necessary
         * for the internal structure of the graph.
         *
         * @note
         * - The constructor does not perform any computations or node additions. It merely initializes the empty graph.
         *
         * @see ~ComputeGraph() for the destructor that cleans up the resources used by the graph.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        ComputeGraph() = default;

        /**
         * @brief Destructor for the ComputeGraph class.
         *
         * The destructor ensures that any resources allocated by the `ComputeGraph` object, such as the nodes and
         * their associated data, are properly cleaned up when the object is destroyed. It performs any necessary
         * memory deallocation or resource release.
         *
         * @note
         * - The destructor does not need to manually delete each node in the graph, as they are typically managed
         *   by smart pointers or are otherwise not responsible for memory deallocation.
         *
         * @see ComputeGraph() for the constructor that initializes the graph.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        ~ComputeGraph() = default;

        /// @}

        /// @name Graph Builders
        /// @{

        /**
         * @brief Adds an input node to the computational graph.
         *
         * This method creates a new `InputNode` with the specified shape and gradient requirements and adds it to the
         * `ComputeGraph` object. The newly created node is also added to the `inputNodes` list, and its name is recorded
         * in the `nodeRoster` and `nodeRosterReverse` maps for easy access by name.
         *
         * @param shape The shape of the input tensor for the node.
         * @param requires_grad A boolean indicating whether the input node requires gradients for backpropagation.
         * @param name The name of the input node. If not provided, a default name is generated.
         * @return A pointer to the newly created `InputNode`.
         *
         * @note
         * - If the name is "default", a unique name is generated for the input node using the node's type and a reference counter.
         * - The node is added to both `inputNodes` and `nodes`, ensuring it is part of the overall computational graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * InputNode* input = graph.addInput({3, 3}, true, "input_node");
         * ```
         *
         * @see nodes::io::InputNode for the class definition of the input node.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        InputNode* addInput(const Tensor::shape_type& shape, bool requires_grad = false,
                            const std::string& name = "default");

        /**
         * @brief Adds an input node to the computational graph using a Tensor.
         *
         * This method creates a new `InputNode` using the provided `Tensor` and adds it to the `ComputeGraph` object.
         * The newly created node is added to both the `inputNodes` and `nodes` lists. Its name is stored in the
         * `nodeRoster` and `nodeRosterReverse` maps, allowing for easy lookup by name. If no name is provided, a unique
         * name is generated.
         *
         * @param tensor The `Tensor` object that represents the data for the input node.
         * @param name The name of the input node. If "default", a unique name is generated.
         * @return A pointer to the newly created `InputNode`.
         *
         * @note
         * - If the `name` is "default", a unique name will be generated for the input node by concatenating the node's type
         *   and a reference counter.
         * - The node is added to both `inputNodes` and `nodes`, ensuring it is part of the computational graph.
         *
         * ### Usage Example:
         * ```cpp
         * Tensor input_tensor({3, 3}, true);  // Example tensor
         * ComputeGraph graph;
         * InputNode* input = graph.addInput(input_tensor, "input_node");
         * ```
         *
         * @see nodes::io::InputNode for the class definition of the input node.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        InputNode* addInput(const Tensor& tensor, const std::string& name = "default");

        /**
         * @brief Adds an existing `InputNode` to the computational graph.
         *
         * This method adds an already created `InputNode` to the `ComputeGraph`. The node is pushed into both the `nodes`
         * and `inputNodes` lists. If a name is provided, the node is stored in the `nodeRoster` and `nodeRosterReverse` maps
         * with the given name. If the name is `"default"`, a unique name is generated for the node.
         *
         * @param input A pointer to the existing `InputNode` to be added to the graph.
         * @param name The name of the input node. If "default", a unique name is generated for the node.
         * @return A pointer to the added `InputNode`.
         *
         * @note
         * - If the `name` is `"default"`, a unique name is generated by concatenating the node's type and a reference counter.
         * - The node is added to both the `inputNodes` and `nodes` lists, ensuring it becomes part of the computational graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * InputNode* input = new InputNode({3, 3}, true);
         * graph.addInput(input, "input_node");
         * ```
         *
         * @see nodes::io::InputNode for more details on the input node class.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        InputNode* addInput(InputNode* input, const std::string& name = "default");

        /**
         * @brief Adds an input node to the computation graph and returns a pointer to the newly created InputNode.
         *
         * @param shape A reference to the shape of the input tensor (host-to-device). This defines the dimensions of the tensor associated with the input node.
         * @param data A pointer to the initial data for the input tensor (host-to-device). It can be nullptr if no initial data is provided.
         * @param requires_grad A boolean indicating whether the input tensor should require gradient computation.
         * @param host A boolean indicating whether the tensor data is stored on the host.
         * @param name A string representing the name of the input node. If set to "default", a unique name will be generated.
         *
         * @return A pointer to the newly created InputNode.
         *
         * This function is used to add an input node to the computation graph. Memory management: It dynamically allocates memory for the InputNode using the `new` operator. The caller does not need to free this memory directly as the graph takes ownership of the node. When the graph is destroyed, it should be responsible for deallocating the memory of all its nodes.
         *
         * Exception handling: This function does not explicitly catch exceptions. If memory allocation for the InputNode fails (`new` throws `std::bad_alloc`), the exception will propagate to the caller. Also, if there are issues with the provided `shape` or `data`, the constructor of InputNode may throw relevant exceptions.
         *
         * This function interacts with the `nodes`, `inputNodes`, `nodeRoster`, and `nodeRosterReverse` members of the `ComputeGraph` class to manage the list of nodes and the mapping between node names and pointers.
         *
         * @throws std::bad_alloc If memory allocation for the InputNode fails.
         *
         * @note
         * - Ensure that the `shape` and `data` are valid and compatible with the requirements of the InputNode constructor.
         * - If the `name` is set to "default", a unique name will be generated based on the node type and a reference counter.
         * - The time complexity of this function is O(1) for the node creation and O(log m) for the insertion into the `nodeRoster` map, where m is the number of nodes in the graph.
         *
         * @warning
         * - The caller should not delete the returned pointer as the graph takes ownership of the node.
         *
         * @code
         * ```cpp
         * #include <vector>
         *
         * ComputeGraph graph;
         * shape_type shape = {2, 2};
         * value_type data[] = {1.0f, 2.0f, 3.0f, 4.0f};
         * InputNode* inputNode = graph.addInput(shape, data, true, true, "my_input");
         * ```
         * @endcode
         */
        InputNode* addInput(const Tensor::shape_type& shape, Tensor::value_type* data, bool requires_grad,
                            bool host, const std::string& name = "default");

        /**
         * @brief Adds an InputNode to the ComputeGraph using a std::initializer_list for data and returns a pointer to the created node.
         *
         * @param shape A reference to the shape of the tensor associated with the InputNode (host-to-device). Defines the dimensions of the input tensor.
         * @param data A std::initializer_list containing the initial values for the tensor (host-to-device).
         * @param requires_grad A boolean indicating whether the tensor of the InputNode should require gradient computation.
         * @param name A string representing the name of the InputNode. If set to "default", a unique name will be generated.
         *
         * @return A pointer to the newly created InputNode.
         *
         * This function is responsible for creating and adding an InputNode to the ComputeGraph. Memory management: It uses the `new` operator to allocate memory for the InputNode, and the ComputeGraph takes ownership of this memory. The memory should be deallocated when the ComputeGraph is destroyed.
         *
         * Exception handling: This function does not explicitly catch exceptions. If memory allocation for the InputNode fails (`new` throws `std::bad_alloc`), or if there are issues with the `shape` or `data` passed to the InputNode constructor, the exceptions will propagate to the caller.
         *
         * It interacts with the `nodes`, `inputNodes`, `nodeRoster`, and `nodeRosterReverse` members of the ComputeGraph to manage the list of nodes and the mapping between node names and pointers.
         *
         * @throws std::bad_alloc If memory allocation for the InputNode fails.
         *
         * @note
         * - Ensure that the `shape` and `data` are compatible with the requirements of the InputNode constructor.
         * - If the `name` is "default", a unique name will be generated based on the node type and a reference counter.
         * - The time complexity of this function is O(1) for node creation and O(log m) for insertion into the `nodeRoster` map, where m is the number of nodes in the graph.
         *
         * @warning
         * - Do not delete the returned pointer as the ComputeGraph takes ownership of the InputNode.
         *
         * @code
         * ```cpp
         * #include <vector>
         *
         * ComputeGraph graph;
         * shape_type shape = {2, 2};
         * InputNode* inputNode = graph.addInput(shape, {1.0f, 2.0f, 3.0f, 4.0f}, true, "my_input");
         * ```
         * @endcode
         */
        InputNode* addInput(const Tensor::shape_type& shape, const std::initializer_list<Tensor::value_type>& data,
                            bool requires_grad, const std::string& name = "default");

        /**
         * @brief Adds a node of any type to the computational graph.
         *
         * This template method allows adding a node of any type derived from the `Node` class to the computational graph.
         * The node is added to the `nodes` list and optionally assigned a name. If the name is `"default"`, a unique
         * name is generated using the node's type and a reference counter.
         *
         * @tparam NodeType The type of the node, which must be derived from `Node`.
         * This allows the method to work with different types of nodes (e.g., `InputNode`, `OutputNode`, etc.).
         *
         * @param node A pointer to the node to be added to the graph.
         * @param name The name of the node. If `"default"`, a unique name will be generated. The default value is `"default"`.
         *
         * @return A pointer to the added node.
         *
         * @note
         * - If the `name` is `"default"`, the method generates a unique name for the node by concatenating the node's type
         * and a reference counter (e.g., `Input_1`, `Add_2`).
         * - The node is added to both the `nodes` list (for graph traversal) and the `nodeRoster`/`nodeRosterReverse` maps
         * (for name-based access).
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * auto* node = new AddNode(&input1, &input2);
         * graph.addNode(node, "Add");
         * ```
         *
         * @see nodes::Node for the base node class.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        template <typename NodeType>
        NodeType* addNode(NodeType* node, const std::string& name = "default") {
            nodes.push_back(node);
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

        /**
         * @brief Adds a node to the computational graph based on the provided node type and inputs.
         *
         * This method adds a node to the computational graph based on the specified node type (e.g., "Input", "Output",
         * "Add", "MatMul", etc.). It also ensures that the required input nodes are present in the graph. Depending on the
         * node type, the method creates and adds the corresponding node to the graph and returns a pointer to the added node.
         * If the node type or input nodes are invalid, an exception will be thrown.
         *
         * @param type The type of the node to be added. It can be one of the following:
         *             "Input", "Output", "Add", "MatMul", "Sub", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Swish",
         *             "ELU", "HardSigmoid", "HardSwish", "Softmax", "MeanSquaredError", "BinaryCrossEntropy",
         *             "ScalarAdd", "ScalarSub", "ScalarMul", "ScalarDiv".
         * @param input1 The name of the first input node. The exact meaning depends on the node type.
         * @param input2 The name of the second input node (if required by the node type).
         * @param name The name of the node to be added. If `"default"`, a unique name is generated. The default value is `"default"`.
         * @param args Additional arguments required for some node types (e.g., parameters for LeakyReLU, ELU, etc.).
         *
         * @return A pointer to the newly added node.
         *
         * @throws std::runtime_error If any required input node is not found, or if an unsupported node type is provided.
         *
         * @note
         * - For "Input" and "Scalar" nodes, the method will print a warning and return `nullptr` because these nodes cannot be added using this method.
         * - If the specified node type requires specific input nodes, the method checks if the input nodes exist in the graph before proceeding.
         * - The method supports variable arguments (`Args... args`) for nodes like LeakyReLU, ELU, and others that may require additional parameters.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * Node* addNode = graph.addNode("Add", "input1", "input2", "add_node");
         * if (addNode != nullptr) {
         *     // Use the addNode pointer here
         * }
         * ```
         *
         * @see nodes::Node for the base class of all node types.
         * @see nodes::io::OutputNode, nodes::calc::AddNode, nodes::calc::MatMulNode, etc., for the specific node classes that are created.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        template <typename... Args>
        Node* addNode(const std::string& type, const std::string& input1, const std::string& input2,
                      const std::string& name = "default", Args... args) {
            if (type == "Input") {
                WARN("Input node cannot be added by addNode(...)");
                return nullptr;
            }
            if (type == "Output") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addOutput(new OutputNode(nodeRoster[input1]), name);
            }
            if (type == "Add") {
                if (nodeRoster.find(input1) == nodeRoster.end() || nodeRoster.find(input2) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new AddNode(nodeRoster[input1], nodeRoster[input2]), name);
            }
            if (type == "MatMul") {
                if (nodeRoster.find(input1) == nodeRoster.end() || nodeRoster.find(input2) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new MatMulNode(nodeRoster[input1], nodeRoster[input2]), name);
            }
            if (type == "Sub") {
                if (nodeRoster.find(input1) == nodeRoster.end() || nodeRoster.find(input2) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new SubNode(nodeRoster[input1], nodeRoster[input2]), name);
            }
            if (type == "ReLU") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new ReLUNode(nodeRoster[input1]), name);
            }
            if (type == "Sigmoid") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new SigmoidNode(nodeRoster[input1]), name);
            }
            if (type == "Tanh") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new TanhNode(nodeRoster[input1]), name);
            }
            if (type == "LeakyReLU") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new LeakyReLUNode(nodeRoster[input1], args...), name);
            }
            if (type == "Swish") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new SwishNode(nodeRoster[input1]), name);
            }
            if (type == "ELU") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new ELUNode(nodeRoster[input1], args...), name);
            }
            if (type == "HardSigmoid") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new HardSigmoidNode(nodeRoster[input1], args...), name);
            }
            if (type == "HardSwish") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new HardSwishNode(nodeRoster[input1], args...), name);
            }
            if (type == "Softmax") {
                if (nodeRoster.find(input1) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addNode(new SoftmaxNode(nodeRoster[input1]), name);
            }
            if (type == "MeanSquaredError") {
                if (nodeRoster.find(input1) == nodeRoster.end() || nodeRoster.find(input2) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addOutput(new MeanSquaredErrorNode(nodeRoster[input1], nodeRoster[input2]), name);
            }
            if (type == "BinaryCrossEntropy") {
                if (nodeRoster.find(input1) == nodeRoster.end() || nodeRoster.find(input2) == nodeRoster.end()) {
                    throw std::runtime_error("Input node not found");
                }
                return addOutput(new BinaryCrossEntropyNode(nodeRoster[input1], nodeRoster[input2]), name);
            }
            if (type == "ScalarAdd" || type == "ScalarSub" || type == "ScalarMul" || type == "ScalarDiv") {
                WARN("Scalar nodes cannot be added by addNode(...)");
                return nullptr;
            }
            throw std::runtime_error("Node type not found");
        }

        /**
         * @brief Adds an output node to the computational graph.
         *
         * This method adds an `OutputNode` to the computational graph. It takes an `OutputNode` pointer and an optional
         * name. The node is added to both the `nodes` list and the `outputNodes` list. If the name is `"default"`, a
         * unique name is generated using the node's type and a reference counter.
         *
         * @param node A pointer to the `OutputNode` to be added to the graph.
         * @param name The name of the node. If `"default"`, a unique name will be generated. The default value is `"default"`.
         *
         * @return A pointer to the added `OutputNode`.
         *
         * @note
         * - If the `name` is `"default"`, the method generates a unique name for the node by concatenating the node's type
         * and a reference counter (e.g., `Output_1`, `Output_2`).
         * - The node is added to both the `nodes` list and the `outputNodes` list for graph traversal and output handling.
         * - The node's name is also stored in the `nodeRoster` and `nodeRosterReverse` maps.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         *
         * // Add basic output node
         * auto* outputNode = new OutputNode(inputNode); // assuming inputNode is a valid InputNode pointer
         * graph.addOutput(outputNode, "output");
         *
         * // Add loss function node
         * auto outputNode = new MeanSquaredErrorNode(inputNode1, inputNode2);
         * graph.addOutput(outputNode, "output");
         * ```
         *
         * @see nodes::io::OutputNode for the output node class.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        OutputNode* addOutput(OutputNode* node, const std::string& name = "default");

        /// @}

        /// @name Modifiers
        /// @{

        /**
         * @brief Performs topological sorting on the computational graph.
         *
         * This function performs topological sorting on the computational graph to order the nodes such that each node
         * appears before any nodes that depend on it. The sorted nodes are stored in the `sortedNodes` vector, which
         * allows for a correct computation order during graph traversal. It uses Kahn's algorithm for topological sorting.
         *
         * @throws std::runtime_error If the graph contains a cycle, indicating that topological sorting is not possible.
         *
         * This method modifies the following member variables:
         * - `sortedNodes`: A vector that stores the nodes in topologically sorted order.
         * - `inDegree`: A map that keeps track of the in-degree (number of incoming edges) for each node.
         * - `adjList`: A map that stores the adjacency list for each node, representing which nodes depend on it.
         *
         * @note
         * - The function assumes that the graph is a Directed Acyclic Graph (DAG). If a cycle is detected during the
         *   sorting process, an exception will be thrown.
         * - This method is useful in scenarios like forward propagation in neural networks, where nodes need to be
         *   processed in a specific order.
         *
         * ### Algorithm Explanation:
         * 1. Initialize the `inDegree` of each node to 0.
         * 2. Build the `adjList` for each node and increment the `inDegree` of nodes that have incoming edges.
         * 3. Initialize a queue with all nodes that have an in-degree of 0 (i.e., no dependencies).
         * 4. Process each node from the queue, adding it to the `sortedNodes` list, and decrement the `inDegree`
         *    of its adjacent nodes (i.e., nodes that depend on it). If any adjacent node's in-degree becomes 0,
         *    it is added to the queue.
         * 5. If the number of nodes in `sortedNodes` does not match the total number of nodes in the graph, a cycle
         *    is detected and an exception is thrown.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Add nodes and edges to the graph...
         * graph.topologicalSort();  // Perform topological sorting
         * ```
         *
         * @see ComputeGraph for more details on the graph structure and node management.
         * @see Node for information on individual node types and their operations.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void topologicalSort();

        /**
         * @brief Resets the gradients of all nodes in the computational graph.
         *
         * This method iterates over all nodes in the computational graph and calls the `zeroGrad()` method on each node's
         * output tensor to reset its gradient. This is useful to clear the gradients between different backward passes,
         * ensuring that previous gradient values do not accumulate. Typically called at the beginning of each new
         * backward pass to prepare the graph for gradient computation.
         *
         * @param None
         *
         * @return None
         *
         * @note
         * - This method assumes that each node has an associated output tensor with a `zeroGrad()` method to reset
         *   gradients.
         * - It does not perform any checks on whether the graph is sorted or whether backward propagation has been
         *   performed previously. It simply clears the gradients of all nodes in the graph.
         * - The method is typically used in training loops to avoid gradient accumulation across iterations.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * graph.zeroGrad();  // Clears the gradients of all nodes in the graph
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void zeroGrad() const;

        /**
         * @brief Randomizes the output tensor of a specified node in the computational graph.
         *
         * This method sets the values of the specified node's output tensor to random values using the provided random
         * seed. The method first checks if the node with the given name exists in the graph. If the node exists, it calls
         * the `randomize()` method on the node’s output tensor. If the node is not found in the graph, a runtime error is
         * thrown.
         *
         * @param name The name of the node whose output tensor should be randomized.
         * @param seed The seed value for the random number generator. If not provided, the seed defaults to 0.
         *
         * @throw std::runtime_error If the node with the given name is not found in the graph.
         *
         * @note
         * - The `randomize()` method is expected to be defined for the node's output tensor to set its values randomly.
         * - The method uses the provided `seed` value to ensure reproducibility of the randomization process.
         * - If the node is not found, an error is thrown to inform the user that the node is missing from the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Assuming "input_node" is a valid node name in the graph
         * graph.randomize("input_node", 42);  // Randomizes the output of "input_node" using seed 42
         * ```
         *
         * @see Tensor::randomize() for the method that randomizes a tensor's data.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void randomize(const std::string& name, unsigned long long seed = 0);

        /**
         * @brief Randomizes the output tensor of a specified node in the computational graph.
         *
         * This method sets the values of the specified node's output tensor to random values using the provided random
         * seed. The method first checks if the given node exists in the graph by searching for it in the list of nodes.
         * If the node exists, it calls the `randomize()` method on the node’s output tensor. If the node is not found in
         * the graph, a runtime error is thrown.
         *
         * @param node A pointer to the `Node` whose output tensor should be randomized.
         * @param seed The seed value for the random number generator.
         *
         * @throw std::runtime_error If the node is not found in the graph.
         *
         * @note
         * - The `randomize()` method is expected to be defined for the node's output tensor to set its values randomly.
         * - The method uses the provided `seed` value to ensure reproducibility of the randomization process.
         * - The node is searched in the `nodes` list to ensure it is part of the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Assuming "inputNode" is a valid pointer to a node in the graph
         * graph.randomize(inputNode, 42);  // Randomizes the output of "inputNode" using seed 42
         * ```
         *
         * @see Tensor::randomize() for the method that randomizes a specific node’s output tensor.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void randomize(const Node* node, unsigned long long seed = 0);

        /**
         * @brief Randomizes the output tensors of all input nodes in the computational graph.
         *
         * This method iterates over all input nodes in the graph and randomizes the output tensor for each of them.
         * It uses the current system time (in nanoseconds) as the seed for the random number generator. Each input node
         * is assigned a unique seed by incrementing the base seed for each randomization.
         *
         * @note
         * - The randomization process is applied to each input node's output tensor.
         * - The seed for randomization is based on the system's current time, ensuring a unique starting point.
         * - The seed is incremented for each input node to provide a different randomization for each one.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Assuming graph has input nodes added
         * graph.randomizeAll();  // Randomizes the output of all input nodes
         * ```
         *
         * @see Tensor::randomize() for the method that randomizes a specific node’s output tensor.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void randomizeAll() const;

        /**
         * @brief Fills the output tensor of a specified node with a given value.
         *
         * This method fills the output tensor of a node, identified by its name, with a specified value. It looks up
         * the node by name in the `nodeRoster`. If the node is found, it calls the `fill` method on the node's output
         * tensor, setting all its elements to the provided value. If the node is not found, an exception is thrown.
         *
         * @param name The name of the node whose output tensor will be filled.
         * @param val The value to fill the output tensor with.
         *
         * @throws std::runtime_error if the node with the specified name is not found in the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Assuming graph has nodes added and a node named "input_1"
         * graph.fill("input_1", 0.0);  // Fills the output tensor of "input_1" with 0.0
         * ```
         *
         * @see Tensor::fill() for the method that fills the tensor with a specific value.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void fill(const std::string& name, Tensor::value_type val);

        /**
         * @brief Fills the output tensor of a specified node with a given value.
         *
         * This method fills the output tensor of a node, identified by its pointer, with a specified value. It checks
         * if the node is present in the graph by searching the node pointer in the `nodes` list. If the node is found,
         * it calls the `fill` method on the node's output tensor to set all its elements to the provided value. If the
         * node is not found in the graph, an exception is thrown.
         *
         * @param node A pointer to the node whose output tensor will be filled.
         * @param val The value to fill the output tensor with.
         *
         * @throws std::runtime_error if the node is not found in the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Assuming graph has nodes added and a valid node pointer "inputNode"
         * graph.fill(inputNode, 0.0);  // Fills the output tensor of "inputNode" with 0.0
         * ```
         *
         * @see Tensor::fill() for the method that fills the tensor with a specific value.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void fill(const Node* node, Tensor::value_type val);

        /**
         * @brief Fills the output tensors of all input nodes with a given value.
         *
         * This method iterates over all input nodes in the computational graph and fills their output tensors with the
         * specified value. It calls the `fill` method on each input node's output tensor to set all its elements to the
         * provided value. This operation is performed for every input node in the graph.
         *
         * @param val The value to fill the output tensors of all input nodes with.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Assuming graph has input nodes added
         * graph.fillAll(0.0);  // Fills the output tensors of all input nodes with 0.0
         * ```
         *
         * @see Tensor::fill() for the method that fills the tensor of a specific node with a value.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void fillAll(Tensor::value_type val) const;

        /**
         * @brief Sets the input data for a specified node in the computational graph.
         *
         * This method sets the input data for a node in the computational graph by copying the provided raw data
         * into the node's output tensor. The input data is assumed to be an array of type `Tensor::value_type` and
         * will be copied into the output tensor of the node specified by the `name`. The shape of the output tensor
         * will be used to determine the amount of data to copy.
         *
         * @param name The name of the node whose input data is to be set.
         * @param data A pointer to the raw input data that will be copied into the node's output tensor.
         *
         * @throws std::runtime_error If the node with the specified `name` is not found in the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * Tensor::value_type inputData[] = {1.0, 2.0, 3.0};  // Example input data
         * graph.setInput("input_node_name", inputData);  // Sets the input data for the specified node
         * ```
         *
         * @see Tensor::dataInject() for the method that inject the data into the tensor.
         * @see data::Tensor for the class representing tensors and their associated operations.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void setInput(const std::string& name, Tensor::value_type* data);

        /**
         * @brief Sets the input data for a specified node in the computational graph using a node pointer.
         *
         * This method sets the input data for a node in the computational graph by copying the provided raw data
         * into the node's output tensor. The input data is assumed to be an array of type `Tensor::value_type` and
         * will be copied into the output tensor of the specified node. The shape of the output tensor will be used
         * to determine the amount of data to copy.
         *
         * @param node A pointer to the `Node` whose input data is to be set.
         * @param data A pointer to the raw input data that will be copied into the node's output tensor.
         *
         * @throws std::runtime_error If the node is not found in the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * Tensor::value_type inputData[] = {1.0, 2.0, 3.0};  // Example input data
         * graph.setInput(inputNode, inputData);  // Sets the input data for the specified node
         * ```
         *
         * @see Tensor::dataInject() for the method that inject the data into the tensor.
         * @see data::Tensor for the class representing tensors and their associated operations.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void setInput(const Node* node, Tensor::value_type* data);

        /// @}

        /// @name Getters
        /// @{

        /**
         * @brief Checks whether the computational graph has been topologically sorted.
         *
         * This function checks if the `sortedNodes` vector contains the nodes in a valid topologically sorted order.
         * It returns `true` if the graph is sorted, meaning that each node appears before any node that depends on it.
         * Otherwise, it returns `false`, indicating that the graph is not sorted.
         *
         * @returns `true` if the graph is sorted, `false` if not.
         *
         * @note
         * - This function does not modify the state of the graph.
         * - It is a helper function that can be used to verify whether a graph needs sorting before traversing.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Add nodes to the graph...
         * if (!graph.isSorted()) {
         *     graph.topologicalSort();  // Sort the graph if it is not sorted
         * }
         * ```
         *
         * @see topologicalSort() for the function that performs the sorting.
         * @see ComputeGraph for more details on the graph structure and node management.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        [[nodiscard]] bool isSorted() const;

        /**
         * @brief Retrieves the output data of the first output node in the computational graph.
         *
         * This method retrieves a pointer to the output data of the first `OutputNode` in the computational graph.
         * The output data is stored in the output tensor of the node. It is important to note that the returned pointer
         * points to data that resides in GPU memory.
         *
         * If no output nodes exist in the graph, a `std::runtime_error` is thrown. The method assumes that there is at least
         * one output node in the graph, and will not return a `nullptr`.
         *
         * @return A pointer to the output data of the first output node in the graph, which is stored in GPU memory.
         *
         * @throws std::runtime_error If no output nodes are present in the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * try {
         *     Tensor::value_type* outputData = graph.getOutput();
         *     // Use the outputData pointer here
         * } catch (const std::runtime_error& e) {
         *     // Handle the case when no output node is present
         *     std::cerr << "Error: " << e.what() << std::endl;
         * }
         * ```
         *
         * @see nodes::io::OutputNode for the output node class.
         * @see data::Tensor for the class representing tensors and their associated operations.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        [[nodiscard]] Tensor::value_type* getOutput() const;

        /**
         * @brief Retrieves the output data of the first output node in the computational graph and copies it to host memory.
         *
         * This method retrieves a pointer to the output data of the first `OutputNode` in the computational graph.
         * It then copies the data from GPU memory to host memory. The returned pointer points to a memory block in
         * host memory that contains the output data.
         *
         * If the graph contains no output nodes, a runtime error is thrown. The method assumes that there is at least
         * one output node in the graph; otherwise, it will throw an exception.
         *
         * The returned pointer points to memory allocated in the host's memory space. The caller is responsible for
         * freeing this memory using `free()` once it's done using the data.
         *
         * @return A pointer to the output data of the first output node in the graph, stored in host memory.
         *         The memory must be freed by the caller after use.
         *
         * @throws std::runtime_error If no output nodes are present in the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * try {
         *     Tensor::value_type* outputDataHost = graph.getOutputHost();
         *     // Use the outputDataHost pointer here
         *     // Remember to free the memory when done
         *     free(outputDataHost);
         * } catch (const std::runtime_error& e) {
         *     std::cerr << "Error: " << e.what() << std::endl;
         * }
         * ```
         *
         * @see nodes::io::OutputNode for the output node class.
         * @see data::Tensor for the class representing tensors and their associated operations.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        [[nodiscard]] Tensor::value_type* getOutputHost() const;

        /**
         * @brief Retrieves the first output node in the computational graph.
         *
         * This method retrieves the first `OutputNode` in the computational graph. The method assumes that there is at least
         * one output node in the graph. If no output nodes exist, a `std::runtime_error` is thrown.
         *
         * @return A pointer to the first output node in the graph.
         *
         * @throws std::runtime_error If no output nodes are present in the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * try {
         *     OutputNode* outputNode = graph.getOutputNode();
         *     // Use the outputNode pointer here
         * } catch (const std::runtime_error& e) {
         *     // Handle the case when no output node is present
         *     std::cerr << "Error: " << e.what() << std::endl;
         * }
         * ```
         *
         * @see nodes::io::OutputNode for the output node class.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        [[nodiscard]] OutputNode* getOutputNode() const;

        /**
         * @brief Retrieves the loss value from the first output node in the computational graph.
         *
         * This method retrieves the loss value computed by the first `OutputNode` in the computational graph. The method
         * assumes that there is at least one output node in the graph. If no output nodes exist, a `std::runtime_error`
         * is thrown.
         *
         * @return The loss value computed by the first output node in the graph.
         *
         * @throws std::runtime_error If no output nodes are present in the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * try {
         *     Tensor::value_type loss = graph.getLoss();
         *     std::cout << "Loss: " << loss << std::endl;
         * } catch (const std::runtime_error& e) {
         *     // Handle the case when no output node is present
         *     std::cerr << "Error: " << e.what() << std::endl;
         * }
         * ```
         *
         * @see nodes::io::OutputNode for the output node class.
         * @see OutputNode::getLoss() for the method in the `OutputNode` class that computes the loss.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        [[nodiscard]] Tensor::value_type getLoss() const;

        /**
         * @brief Prints the details of the computational graph to the provided output stream.
         *
         * The `print` method prints a detailed description of the computational graph, including each node's name,
         * its preceding (input) nodes, following (output) nodes, data, and gradients. If the graph is not sorted,
         * it will automatically perform a topological sort before printing the details. The method assumes that
         * the graph contains at least one output node and prints the loss value of the first output node.
         *
         * @param os The output stream where the graph details will be printed (e.g., `std::cout`).
         * @return The same output stream after printing the graph details, enabling method chaining.
         *
         * @throws std::runtime_error if an error occurs during the process.
         *
         * @note
         * - If the graph is not sorted, the method will automatically call `topologicalSort()` to sort the nodes.
         * - The method prints the loss value of the first output node in the graph, assuming there is at least one output node.
         *
         * ### Example:
         * ```cpp
         * ComputeGraph graph;
         * // Add nodes and build the graph
         * graph.print(std::cout);  // Print the graph details to the console
         * ```
         *
         * @see topologicalSort() for sorting the nodes of the graph.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        std::ostream& print(std::ostream& os);

        /**
         * @brief Retrieves the node associated with the given name in the computational graph.
         *
         * This method overloads the `operator[]` to provide access to nodes in the computational graph by their name.
         * It allows for easy retrieval of nodes from the `nodeRoster` map. The operator returns a pointer to the node
         * associated with the provided name.
         *
         * If the node with the specified name is not found, the method will cause undefined behavior as it directly
         * accesses the `nodeRoster` map without checking for the node's existence.
         *
         * @param name The name of the node to retrieve.
         *
         * @return A pointer to the node associated with the specified name.
         *
         * @note
         * - If the node does not exist in `nodeRoster`, this method will cause undefined behavior because it directly
         *   accesses the map. To safely check for the existence of a node, consider using `find()` instead.
         * - This operator does not throw exceptions; it relies on the `nodeRoster` map's behavior when accessing an element by key.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * Node* node = graph["node_name"];
         * if (node != nullptr) {
         *     // Use the node here
         * } else {
         *     // Handle the case when the node is not found
         * }
         * ```
         *
         * @see nodeRoster for the map storing nodes by name.
         * @see Node for the base class of all nodes.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        Node* operator[](const std::string& name);

        /**
         * @brief Generates a formatted string representing the list of nodes in the ComputeGraph.
         *
         * @return A string containing a tabular representation of node names and types.
         *
         * This function iterates through the `nodeRoster` of the ComputeGraph to determine the maximum width required for displaying node names and types. It then constructs a formatted string using `std::ostringstream` to present the nodes in a tabular format.
         *
         * Memory management: The function does not allocate any dynamic memory that needs to be explicitly managed. It uses local variables and the `std::ostringstream` which handles its own memory internally.
         *
         * Exception handling: This function does not explicitly catch exceptions. Exceptions such as `std::bad_alloc` may be thrown if there is insufficient memory during the construction of the output string.
         *
         * This function only depends on the `nodeRoster` member of the `ComputeGraph` class, which stores the mapping between node names and pointers.
         *
         * @throws std::bad_alloc If there is insufficient memory to construct the output string.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of nodes in the `nodeRoster`, as it iterates through the map twice.
         * - The output string is formatted in a left - aligned tabular style.
         *
         * @code
         * ```cpp
         * ComputeGraph graph;
         * // Assume some nodes are added to the graph
         * std::string nodesListStr = graph.nodesList();
         * std::cout << nodesListStr << std::endl;
         * ```
         * @endcode
         */
        std::string nodesList();

        /// @}

        /// @name Computation
        /// @{

        /**
         * @brief Performs forward propagation on the computational graph.
         *
         * This method performs forward propagation on all nodes in the computational graph. It first ensures the graph
         * is sorted in topological order (if not already sorted), and then propagates the data through each node in
         * the sorted order. Each node's `forward()` method is called to compute its output based on its inputs.
         *
         * @param None
         *
         * @return None
         *
         * @note
         * - This method checks if the graph is sorted by calling the `isSorted()` method. If the graph is not sorted,
         *   it calls the `topologicalSort()` method to sort the nodes in topological order before performing the forward propagation.
         * - The nodes are processed in sorted order, ensuring that each node’s inputs are computed before the node itself.
         * - After calling `topologicalSort()`, the `forward()` method calls each node's `forward()` method to compute the
         *   node’s output and propagate the result through the graph.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Assuming nodes are added to the graph...
         * graph.forward(); // Performs forward propagation on all nodes in the graph
         * ```
         *
         * @see topologicalSort() for the method that sorts the nodes in topological order.
         * @see isSorted() for the method that checks if the graph is sorted.
         * @see backward() for the method that performs backward propagation.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void forward();

        /**
         * @brief Performs backward propagation on the computational graph.
         *
         * This method performs backward propagation starting from the output node(s) in the computational graph.
         * The method first checks if the graph is sorted. If the graph is not sorted, a runtime error is thrown. If the
         * graph is sorted, the backward propagation is performed by iterating over the nodes in reverse topological order
         * (i.e., from outputs to inputs). Each node’s `backward()` method is called to compute gradients with respect to
         * its inputs.
         *
         * @param None
         *
         * @return None
         *
         * @throw std::runtime_error If the graph is not sorted or if there are multiple output nodes or no output node.
         *
         * @note
         * - If the graph is not sorted, a runtime error is thrown because backward propagation relies on the topological
         *   order of the nodes. Sorting ensures that each node’s gradient can be computed in the correct order.
         * - If the graph has exactly one output node, the method proceeds with backward propagation in reverse topological
         *   order (from the output node back to the input nodes).
         * - If the graph has no output nodes or multiple output nodes, a runtime error is thrown.
         *
         * ### Why Not Automatically Sort the Graph?
         * The `backward()` method does not automatically sort the graph because backward propagation must correspond to a
         * previously completed forward pass. A forward pass determines the order of operations and ensures that the graph is
         * in a valid state for backward propagation. Automatically sorting the graph would interfere with this flow, as
         * backward propagation is dependent on the state of the graph after forward propagation. Hence, the graph is only
         * processed for backward propagation if it has been sorted and the forward pass has already occurred.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * // Assuming nodes are added and the graph is sorted...
         * graph.forward();  // Perform forward propagation first
         * graph.backward(); // Perform backward propagation after forward pass
         * ```
         *
         * @see isSorted() for the method that checks if the graph is sorted.
         * @see forward() for the method that performs forward propagation.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void backward();

        /**
         * @brief Updates the parameters of the nodes that require gradients using the provided optimizer.
         *
         * This method iterates through all the nodes in the computational graph and applies the optimizer's update
         * step to the nodes that have their `output` tensor marked as requiring gradients. The update is performed by
         * calling the `step` method of the provided optimizer for each node.
         *
         * @param optimizer A pointer to the optimizer that will be used to update the parameters.
         *                  The optimizer's `step` method is called for each node that requires gradients.
         *
         * @throws std::runtime_error If the optimizer is a null pointer.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * Optimizer* optimizer = new SGD(learning_rate);  // assuming an SGD optimizer
         * graph.update(optimizer);
         * ```
         *
         * @see opt::Optimizer for the interface of the optimizer class.
         * @see nodes::Node for the node class that holds the parameters and their gradients.
         * @see data::Tensor for the tensor class associated with the node's output.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void update(Optimizer* optimizer) const;

        bool inGraph(Node* node) const;

        /// @}

        /// @name File Managers
        /// @{

        /**
         * @brief Saves the current computational graph to a JSON file.
         *
         * This method serializes the entire computational graph into a JSON file at the specified path.
         * It traverses the nodes in the graph and stores their types, names, input-output relationships,
         * shapes, data, gradients (if required), and other relevant information in JSON format.
         * The serialized graph can later be loaded for further processing or visualization.
         *
         * ### Graph Serialization:
         * - **Nodes**: Each node's type, name, input-output connections, and other details are stored.
         * - **Pre and Post nodes**: Lists of indices for input (pre) and output (post) nodes are saved.
         * - **Data and Gradients**: Node data and gradients are copied from GPU to host and serialized.
         *
         * ### Error Handling:
         * - Throws a `std::runtime_error` if the path is empty, the graph is not sorted, or if there is any failure during file writing.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * graph.save("path_to_save_graph.json");
         * ```
         *
         * @param path The file path where the graph should be saved.
         *
         * @throws std::runtime_error If the path is empty, the graph is not sorted, or file writing fails.
         *
         * @see nodes::Node for the base class of all nodes.
         * @see data::Tensor for the class representing tensors and their associated operations.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void save(const std::string& path);

        /**
         * @brief Loads a computational graph from a JSON file.
         *
         * This method deserializes the computational graph from the provided JSON file and reconstructs
         * the nodes, their types, names, input-output relationships, shapes, data, gradients, and other
         * relevant information. It validates the file structure and populates the graph accordingly.
         *
         * ### Graph Deserialization:
         * - **Nodes**: Each node's type, name, input-output connections, and other details are extracted.
         * - **Pre and Post nodes**: Lists of indices for input (pre) and output (post) nodes are read.
         * - **Data and Gradients**: Node data and gradients (if required) are read and restored into their respective tensors.
         *
         * ### Error Handling:
         * - Throws a `std::runtime_error` if the path is empty, the graph is already loaded, or there is an issue opening the file.
         *
         * ### Usage Example:
         * ```cpp
         * ComputeGraph graph;
         * graph.load("path_to_load_graph.json");
         * ```
         *
         * @param path The file path from which the graph should be loaded.
         *
         * @throws std::runtime_error If the path is empty, the graph is already loaded, or file reading fails.
         *
         * @see nodes::Node for the base class of all nodes.
         * @see data::Tensor for the class representing tensors and their associated operations.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/09
         */
        void load(const std::string& path);

        /// @}
    };
}

#endif //COMPUTEGRAPH_CUH

/**
 * @file Nodes.cuh
 * @brief Declaration of the `Node` class and various derived node classes for neural network operations.
 *
 * This file provides the declaration of the `Node` class, which serves as an abstract base class for nodes
 * in a neural network or computational graph. It also defines a variety of derived node classes that represent
 * specific layers or operations commonly used in machine learning models, such as activation functions, loss functions,
 * and mathematical operations.
 *
 * @details
 * The `Node` class encapsulates the following key features:
 * - **Forward and Backward Passes**: Defines pure virtual functions `forward()` and `backward()`, which must be
 *   implemented by derived classes to define the specific operations for data propagation and gradient computation.
 * - **Tensor Operations**: Interacts with `Tensor` objects for storing and manipulating data.
 * - **Graph Structure**: Each node has input and output connections, represented by vectors of pointers to other nodes and
 *   shared pointers to tensors, respectively.
 *
 * The file also defines several derived classes for specific operations, such as:
 * - **Activation Functions**: `LeakyReLUNode`, `SwishNode`, `ELUNode`, `HardSigmoidNode`, `HardSwishNode`, and `SoftmaxNode`.
 * - **Mathematical Operations**: `AddNode`, `MatMulNode`, `ScalarMulNode`, `ScalarDivNode`, etc.
 * - **Loss Functions**: `MeanSquaredErrorNode`, `BinaryCrossEntropyNode`, which are used for computing the error during training.
 *
 * These classes implement the `forward()` and `backward()` methods to perform specific operations and propagate
 * gradients during the training process of the neural network.
 *
 * This class is part of the `nz::nodes` namespace, and each derived class represents a specific computational layer
 * or operation used in deep learning models.
 *
 * @note
 * - Ensure that proper memory management is applied when using these node classes, particularly when dealing with
 *   GPU memory and tensor data.
 *
 * @author
 * Mgepahmge (https://github.com/Mgepahmge)
 *
 * @date
 * 2024/11/29
 */


#ifndef NODES_CUH
#define NODES_CUH

#include <memory>
#include "Tensor.cuh"

/**
 * @namespace nz::nodes
 * @brief Contains classes and functionality for nodes in a neural network or computational graph.
 *
 * The `nz::nodes` namespace provides a collection of classes that represent various layers and operations
 * in a neural network. Each node is an essential component of a computational graph, responsible for performing
 * specific computations during the forward and backward passes.
 *
 * This namespace includes:
 * - **Node Class**: The abstract base class `Node`, which defines the interface for all types of nodes. It provides
 *   the basic structure and functionality, including methods for forward and backward passes.
 * - **Derived Node Classes**: A set of derived classes representing common operations and layers in neural networks,
 *   including activation functions, mathematical operations, and loss functions. Examples include:
 *   - **Activation Functions**: `LeakyReLUNode`, `SwishNode`, `ELUNode`, `HardSigmoidNode`, `HardSwishNode`, `SoftmaxNode`.
 *   - **Mathematical Operations**: `AddNode`, `MatMulNode`, `ScalarMulNode`, `ScalarDivNode`, etc.
 *   - **Loss Functions**: `MeanSquaredErrorNode`, `BinaryCrossEntropyNode`.
 *
 * The nodes in this namespace work with `Tensor` objects to propagate data and gradients through the network,
 * supporting the training and inference processes of deep learning models.
 *
 * @note
 * - The nodes in this namespace are designed to be used as part of a computational graph, and each node can be
 *   connected to other nodes to define the structure of a neural network.
 * - Ensure proper memory management when working with tensors, particularly when dealing with GPU memory.
 *
 * @author
 * Mgepahmge (https://github.com/Mgepahmge)
 *
 * @date
 * 2024/11/29
 */
namespace nz::nodes {
    using namespace data;

    /**
     * @class Node
     * @brief Base class for nodes in a neural network or computational graph.
     *
     * The `Node` class serves as an abstract base class for all types of nodes in a computational graph, commonly
     * used in neural networks. Each node represents an operation or a layer in the graph, with input and output
     * connections that allow data to flow through the network. The `forward()` and `backward()` methods define the
     * computations to be performed during the forward and backward passes of the network, respectively.
     *
     * This class is designed to be subclassed and extended for specific layers or operations. Derived classes are
     * required to implement the `forward()` and `backward()` methods to define the specific computations for each node.
     *
     * @details
     * Key features:
     * - **Inputs**: A vector of pointers to other nodes that provide input data to this node.
     * - **Output**: A shared pointer to a `Tensor` object that stores the result of this node's computation.
     * - **Type**: A string indicating the type of the node (e.g., "Basic", "Input", "MatMul").
     * - **Forward and Backward Passes**: The pure virtual functions `forward()` and `backward()` that must be implemented
     *   by derived classes to perform the forward and backward propagation steps of the neural network.
     *
     * This class is part of the `nz::nodes` namespace, and is intended to be used as a base class for defining
     * custom layers or operations in a neural network.
     *
     * @note
     * - Derived classes must implement the `forward()` and `backward()` functions to define the specific computations for the node.
     * - This class is designed to be used within a larger computational graph, where nodes are connected to form a complete neural network.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/11/29
     */
    class DL_API Node {
    public:
        Node() = default;
        virtual ~Node() = default;
        std::vector<Node*> inputs;
        std::shared_ptr<Tensor> output;
        std::string type = "Basic";

        /**
         * @brief Abstract method for the forward pass computation.
         *
         * The `forward()` method is a pure virtual function in the `Node` class, which must be implemented by derived classes.
         * It is responsible for performing the computation during the forward pass of the neural network or computational graph.
         *
         * In the forward pass, data flows through the network from input nodes to output nodes, and each node performs its
         * specific computation (e.g., activation, matrix multiplication, etc.) based on the data it receives as input.
         *
         * Derived classes that represent specific layers or operations (such as activation functions, convolution layers, etc.)
         * must implement this method to define the exact computation to be performed for that layer.
         *
         * @note
         * - The `forward()` method must be implemented by any class derived from `Node`. It should modify the output of the
         *   node based on its inputs and computation.
         * - This method does not return any value, as it updates the node's output tensor directly.
         *
         * @see backward() for the reverse propagation (gradient calculation) method.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/11/29
         */
        virtual void forward() = 0;

        /**
         * @brief Abstract method for the backward pass (gradient computation).
         *
         * The `backward()` method is a pure virtual function in the `Node` class, which must be implemented by derived classes.
         * It is responsible for computing the gradients during the backward pass of the neural network or computational graph,
         * which is used for backpropagation in training.
         *
         * During the backward pass, the error gradients are propagated backward through the network, from the output nodes
         * to the input nodes. Each node computes the gradient of its output with respect to its input, using the chain rule of
         * calculus, to update the weights or parameters of the network.
         *
         * Derived classes that represent specific layers or operations must implement this method to define how gradients
         * are calculated for that particular layer or operation.
         *
         * @note
         * - The `backward()` method must be implemented by any class derived from `Node`. It should compute the gradient of
         *   the output with respect to the node's input and store it in the node's `grad` tensor.
         * - This method is essential for the backpropagation process during training, allowing the model to adjust its parameters
         *   based on the computed gradients.
         *
         * @see forward() for the forward propagation (computation) method.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/11/29
         */
        virtual void backward() = 0;

        /**
         * @brief Prints the type, data, and gradient of the node.
         *
         * The `print()` method outputs the information about the node, including its type, the tensor data stored in the node's
         * output, and the corresponding gradient. This is useful for debugging and inspecting the state of nodes in a
         * computational graph or during training, allowing for easy visualization of the node's content and gradients.
         *
         * The method outputs the following details:
         * - **Type**: The type of the node (e.g., the operation it represents, such as "MatrixMul", "ReLU", etc.).
         * - **Data**: The tensor data stored in the node's `output` tensor.
         * - **Gradient**: If the node has a computed gradient, it is also displayed, providing insights into the gradient values
         *   that are being backpropagated through the network during training.
         *
         * This method is primarily used for debugging and monitoring the state of tensors and gradients, making it easier
         * to inspect how the data and gradients flow through the network.
         *
         * @note
         * - The `output` tensor should contain both the data and the gradient information, and both are printed when this
         *   method is called.
         * - This method is typically used during development or debugging phases and should not be used in performance-critical
         *   code as it involves printing potentially large amounts of data.
         *
         * @param os The output stream (e.g., `std::cout`) to which the node's information will be printed.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/11/29
         */
        virtual void print(std::ostream& os) const;

        /**
         * @brief Injects data into a relevant tensor object, optionally setting its gradient requirement.
         *
         * @param data A pointer to the data to be injected into the tensor (host-to-device). This data will be used to populate the tensor.
         * @param grad A boolean indicating whether the tensor should require gradient computation after data injection. Defaults to false.
         *
         * @return None.
         *
         * This function is designed to inject data into a tensor object. Memory management within this function is handled by the underlying tensor operations. It is assumed that the tensor object has already allocated the necessary memory to hold the data pointed to by `data`.
         *
         * Regarding exception handling, this function does not explicitly catch any exceptions. Exceptions that might occur during data injection, such as memory access errors or CUDA errors (if applicable), will propagate to the caller.
         *
         * This function likely interacts with other components related to the tensor, such as the computation graph or the gradient computation system, depending on the value of `grad`.
         *
         * @throws None explicitly, but underlying tensor operations may throw exceptions, such as std::bad_alloc if memory allocation fails during the injection process.
         *
         * @note
         * - Ensure that the `data` pointer is valid and points to enough data to fill the target tensor.
         * - The CUDA runtime environment should be properly initialized before calling this function if the tensor is using CUDA memory.
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor, as it involves copying data into the tensor.
         *
         * @code
         * ```cpp
         *
         * value_type data[] = {1.0f, 2.0f, 3.0f, 4.0f};
         * // Assume there is an object that has the dataInject method
         * InputNode input({2, 2}), true);
         * obj.dataInject(data);
         * ```
         * @endcode
         */
        void dataInject(Tensor::value_type* data, bool grad = false) const;

        /**
         * @brief Injects data from an iterator range into the output tensor of the InputNode, optionally setting its gradient requirement.
         *
         * @tparam Iterator The type of the iterators used to define the data range. It should support the standard iterator operations like dereferencing and incrementing.
         * @param begin An iterator pointing to the beginning of the data range (host-to-device). The data in this range will be injected into the output tensor.
         * @param end An iterator pointing to the end of the data range (host-to-device).
         * @param grad A boolean indicating whether the output tensor should require gradient computation after data injection. Defaults to false.
         *
         * @return None.
         *
         * This template function is used to inject data from an iterator range into the output tensor of the InputNode. Memory management is handled by the underlying `dataInject` method of the `Tensor` class. It is assumed that the `output` tensor has already allocated sufficient memory to hold the data from the iterator range.
         *
         * Regarding exception handling, this function does not explicitly catch any exceptions. Exceptions that might occur during data injection, such as iterator invalidation or memory allocation errors in the `Tensor` class, will propagate to the caller.
         *
         * This function serves as a wrapper around the `dataInject` method of the `output` tensor, facilitating the use of iterators to provide data for injection.
         *
         * @throws None explicitly, but the `dataInject` method of the `Tensor` class may throw exceptions, such as `std::bad_alloc` if memory allocation fails during the injection process.
         *
         * @note
         * - Ensure that the iterator range `[begin, end)` is valid and that the data type pointed to by the iterators is compatible with the `Tensor::value_type`.
         * - The CUDA runtime environment should be properly initialized before calling this function if the tensor is using CUDA memory.
         * - The time complexity of this function is O(n), where n is the number of elements in the iterator range, as it involves copying data from the range into the tensor.
         *
         * @code
         * ```cpp
         * #include <vector>
         *
         * std::vector<value_type> data = {1.0f, 2.0f, 3.0f, 4.0f};
         * InputNode inputNode({2, 2}, true);
         * inputNode.dataInject(data.begin(), data.end());
         * ```
         * @endcode
         */
        template <typename Iterator>
        void dataInject(Iterator begin, Iterator end, const bool grad = false) const {
            output->dataInject(begin, end, grad);
        }

        /**
         * @brief Injects data from a std::initializer_list into the output tensor of the Node, optionally setting its gradient requirement.
         *
         * @param data A std::initializer_list containing the data to be injected into the output tensor (host-to-device).
         * @param grad A boolean indicating whether the output tensor should require gradient computation after data injection.
         *
         * @return None.
         *
         * This function is responsible for injecting data from a std::initializer_list into the output tensor of the Node. Memory management is handled by the underlying `dataInject` method of the `Tensor` class. The `output` tensor is assumed to have already allocated enough memory to accommodate the data in the `std::initializer_list`.
         *
         * Regarding exception handling, this function does not explicitly catch any exceptions. Exceptions that might occur during data injection, such as memory allocation errors in the `Tensor` class, will propagate to the caller.
         *
         * This function acts as a bridge between the Node and its output tensor, allowing data to be easily provided using a `std::initializer_list`.
         *
         * @throws None explicitly, but the `dataInject` method of the `Tensor` class may throw exceptions, such as `std::bad_alloc` if memory allocation fails during the injection process.
         *
         * @note
         * - Ensure that the `std::initializer_list` contains enough elements to fill the output tensor according to its shape.
         * - The CUDA runtime environment should be properly initialized before calling this function if the tensor is using CUDA memory.
         * - The time complexity of this function is O(n), where n is the number of elements in the `std::initializer_list`, as it involves copying data from the list into the tensor.
         *
         * @code
         * ```cpp
         *
         * InputNode node({2, 2}, true);
         * node.dataInject({1.0f, 2.0f, 3.0f, 4.0f});
         * ```
         * @endcode
         */
        void dataInject(const std::initializer_list<Tensor::value_type>& data, bool grad = false) const;
    };

    /**
     * @brief Overloads the `<<` operator to print information about a node.
     *
     * The `operator<<` is overloaded to provide a convenient way to print detailed information about a node, including its
     * type, data, gradient, and loss (if applicable). This operator calls the `print()` method of the node, which handles
     * the actual formatting and output of the node's information.
     *
     * The operator outputs the following details:
     * - **Type**: The type of the node (e.g., the operation it represents, such as "MatrixMul", "ReLU", etc.).
     * - **Data**: The tensor data stored in the node's `output` tensor.
     * - **Gradient**: If the node has a computed gradient, it is displayed, providing insights into the gradient values
     *   being backpropagated during training.
     * - **Loss**: The loss value associated with the node (if applicable), which can be used to monitor the error or
     *   discrepancy in the node during the forward-backward pass.
     *
     * This operator is primarily used for debugging, logging, and inspecting the state of the node, including its tensor data,
     * gradients, and any associated loss. By using the `<<` operator, you can easily print the node's information directly to
     * standard output or any other output stream.
     *
     * @note
     * - The `print()` method must be implemented by the node's class (or any class derived from it). This method should handle
     *   printing the type, data, gradient, and loss for that specific class.
     * - This operator is designed to be used with any class that has a `print()` method, making it a flexible and reusable
     *   solution for logging and debugging.
     *
     * @param os The output stream (e.g., `std::cout`) to which the node's information will be printed.
     * @param node The node object to be printed. It is passed as a const reference to ensure it is not modified.
     *
     * @return The output stream (`os`), allowing the operator to be used in chain expressions like `std::cout << node1 << node2;`.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/11/29
     */
    template <typename T>
    std::enable_if_t<std::is_base_of_v<Node, T>, std::ostream&>
    operator<<(std::ostream& os, const T& node) {
        node.print(os);
        return os;
    }

    /**
     * @namespace nz::nodes::io
     * @brief This namespace contains standard nodes used in computational graphs for neural networks.
     *
     * The `nz::nodes::io` namespace includes the basic building blocks for a computational graph,
     * such as input nodes and output nodes. These nodes serve as the primary interface for data flow in neural networks.
     *
     * This namespace includes the following classes:
     * - **InputNode**: Represents an input node that provides data to the computational graph.
     * - **OutputNode**: A base class for loss function nodes that connects to the output of the computational graph.
     *
     * These nodes are designed to be used in a larger computational graph where data flows from one node to another
     * during the forward and backward passes. The input node holds the input data and passes it forward, while the
     * output node computes the loss and manages the gradient flow during backpropagation.
     *
     * @note
     * - The `InputNode` does not perform computations, only providing data for the graph.
     * - The `OutputNode` serves as the base class for more specific loss functions (e.g., Mean Squared Error, Cross-Entropy).
     *
     * @see InputNode for details on providing data to the network.
     * @see OutputNode for details on loss function nodes.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/11/29
     */
    namespace io {
        /**
        * @class InputNode
        * @brief Represents an input node in a computational graph.
        *
        * The `InputNode` class is a subclass of `Node`, representing a node that holds the input data for a neural network
        * or computational graph. It is designed to store the input tensor and pass it forward through the graph.
        * `InputNode` does not perform any computations in the forward or backward passes; its main role is to provide
        * input data to the network.
        *
        * @details
        * Key features:
        * - **Tensor Output**: The `InputNode` stores a `Tensor` as its output, which is initialized either from a shape
        *   or an existing tensor.
        * - **No Forward or Backward Operations**: The `forward()` and `backward()` methods are implemented as empty,
        *   since this node only provides input data and does not modify the network during these passes.
        * - **Shape and Gradient Support**: The shape of the input tensor and whether it requires gradients is configurable
        *   during initialization.
        *
        * This class is part of the `nz::nodes` namespace and serves as a fundamental part of the computational graph,
        * providing input data to the network during the forward pass.
        *
        * @note
        * - The `forward()` and `backward()` methods are implemented but do not perform any operations for the `InputNode`.
        * - This class is designed to be used as the starting point of a computational graph, where other nodes depend on
        *   the input data provided by this node.
        *
        * ### Usage Example:
        * ```cpp
        * // Example 1: Creating an InputNode with a specific shape
        * Tensor::shape_type shape = {3, 3};  // Define a 3x3 tensor
        * InputNode input_node(shape, true);  // Create an InputNode with shape {3, 3} and requires_grad = true
        * input_node.output->fill(0.5f);  // Fill the input tensor with a value of 0.5
        *
        * // Example 2: Creating an InputNode from an existing tensor
        * Tensor existing_tensor({2, 2});
        * existing_tensor.fill(1.0f);  // Fill the existing tensor with 1.0
        * InputNode input_node_from_tensor(existing_tensor);  // Create an InputNode from the existing tensor
        *
        * // Example 3: Using InputNode in a computational graph
        * InputNode input_node({4, 4});  // Create an InputNode with shape {4, 4}
        * input_node.output->fill(2.0f);  // Fill the tensor with the value 2.0
        *
        * // In the network, this node will pass the data to subsequent nodes during forward propagation.
        * ```
        *
        * @author
        * Mgepahmge (https://github.com/Mgepahmge)
        *
        * @date
        * 2024/11/29
        */
        class DL_API InputNode : public Node {
        public:
            /**
             * @brief Constructor to initialize an `InputNode` with a specified shape and gradient requirement.
             *
             * This constructor initializes an `InputNode` that holds a tensor with the specified shape. The tensor
             * is created with the specified shape, and it can optionally track gradients if `requires_grad` is set to `true`.
             * The `InputNode` does not perform any computations; its primary role is to hold and provide input data to the
             * computational graph.
             *
             * @param shape The shape of the tensor to be stored in the `InputNode`. This defines the dimensions of the input data.
             * @param requires_grad A boolean flag indicating whether the tensor should track gradients. Defaults to `false`.
             *
             * The `InputNode` will store a `Tensor` object, initialized with the given shape and gradient tracking setting.
             * This tensor can then be used as input for subsequent nodes in the computational graph.
             *
             * @note
             * - The `InputNode` class does not perform any computations during the forward or backward passes. It simply
             *   stores and provides input data.
             * - `requires_grad` determines whether the input tensor will store gradients, which is useful if the input data
             *   is part of the network's parameters.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            explicit InputNode(const Tensor::shape_type& shape, bool requires_grad = false);

            /**
             * @brief Constructor to initialize an `InputNode` with an existing `Tensor`.
             *
             * This constructor initializes an `InputNode` using an existing `Tensor` object. The `Tensor` object is directly
             * assigned to the `output` of the `InputNode`, allowing the node to use the provided tensor as its input data.
             * The `InputNode` does not perform any computations but simply holds and provides the given tensor data to the
             * computational graph.
             *
             * @param tensor The existing `Tensor` object to be used as the input for the `InputNode`. This tensor contains
             *               the input data to be passed through the network.
             *
             * The `InputNode` stores the provided tensor in its `output` member, which will be used by other nodes in the graph.
             *
             * @note
             * - The `InputNode` does not modify the given tensor, it simply holds a reference to it.
             * - This constructor is useful when you already have a tensor (e.g., loaded from a file or created elsewhere) and
             *   want to use it as input to the computational graph.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            explicit InputNode(const Tensor& tensor);

            /**
             * @brief Constructs an InputNode object with specified tensor shape, data, gradient requirement, and data location.
             *
             * @param shape A reference to the shape of the output tensor of the input node (host-to-device). It defines the dimensions of the tensor.
             * @param data A pointer to the initial data of the output tensor. The data can be either on the host or device depending on the `host` parameter.
             * @param requires_grad A boolean indicating whether the output tensor requires gradient computation.
             * @param host A boolean indicating whether the data pointed to by `data` is on the host or device. If true, data is on the host; otherwise, it is on the device.
             *
             * @return None. This is a constructor.
             *
             * This constructor initializes an `InputNode` object. It creates a new `Tensor` object using the provided `shape`, `data`, `requires_grad`, and `host` parameters and stores a shared pointer to this tensor in the `output` member variable.
             *
             * In terms of memory management, the `std::shared_ptr` in `output` takes care of the memory of the `Tensor` object. When the last reference to the `Tensor` object held by a `std::shared_ptr` is destroyed, the `Tensor` object will be automatically deleted.
             *
             * Regarding exception handling, this constructor does not explicitly catch any exceptions thrown by the `Tensor` constructor. If the `Tensor` constructor fails (e.g., due to insufficient memory or invalid input), the exception will propagate to the caller.
             *
             * This constructor is a fundamental part of the `InputNode` class as it initializes the output tensor of the input node.
             *
             * @throws None explicitly, but the `Tensor` constructor may throw exceptions, such as `std::bad_alloc` if memory allocation fails.
             *
             * @note
             * - Ensure that the `data` pointer is valid and points to enough data to fill the tensor according to the specified shape.
             * - The CUDA runtime environment should be properly initialized before calling this constructor if the tensor is using CUDA memory.
             * - This constructor has a time complexity of O(1) for creating the `std::shared_ptr` and O(n) for the `Tensor` constructor, where n is the total number of elements in the tensor.
             *
             * @code
             * ```cpp
             * #include <vector>
             *
             * shape_type shape = {2, 3};
             * value_type data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
             * try {
             *     InputNode inputNode(shape, data, true, true);
             * } catch (const std::exception& e) {
             *     std::cerr << e.what() << std::endl;
             * }
             * ```
             * @endcode
             */
            explicit InputNode(const Tensor::shape_type& shape, Tensor::value_type* data,
                               bool requires_grad = false, bool host = false);

            /**
             * @brief Constructs an InputNode object with a specified tensor shape, initializer list data, and gradient requirement.
             *
             * @param shape A reference to the shape of the output tensor of the input node (host-to-device). It determines the dimensions and total size of the tensor.
             * @param data A std::initializer_list containing the initial data for the output tensor (host-to-device).
             * @param requires_grad A boolean indicating whether the output tensor requires gradient computation.
             *
             * @return None. This is a constructor.
             *
             * This constructor initializes an InputNode object. It creates a new Tensor object using the provided shape, initializer list data, and gradient requirement, and stores a shared pointer to this tensor in the output member variable.
             *
             * For memory management, the std::shared_ptr in output takes care of the Tensor object's memory. When the last reference to the Tensor object held by a std::shared_ptr is destroyed, the Tensor object will be automatically deleted.
             *
             * Regarding exception handling, this constructor does not explicitly catch any exceptions thrown by the Tensor constructor. If the Tensor constructor fails (e.g., due to insufficient memory or an invalid initializer list size), the exception will propagate to the caller.
             *
             * This constructor is an important part of the InputNode class as it provides a convenient way to initialize the output tensor of the input node with an initializer list.
             *
             * @throws None explicitly, but the Tensor constructor may throw exceptions such as std::invalid_argument if the initializer list size is insufficient or std::bad_alloc if memory allocation fails.
             *
             * @note
             * - Ensure that the std::initializer_list contains enough elements to fill the tensor according to the specified shape.
             * - The CUDA runtime environment should be properly initialized before calling this constructor if the tensor is using CUDA memory.
             * - The time complexity of this constructor is O(1) for creating the std::shared_ptr and O(n) for the Tensor constructor, where n is the total number of elements in the tensor.
             *
             * @code
             * ```cpp
             * #include <vector>
             * // Assume Tensor::shape_type and Tensor::value_type are defined
             * using shape_type = std::vector<size_t>;
             * using value_type = float;
             *
             * shape_type shape = {2, 3};
             * try {
             *     InputNode inputNode(shape, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
             * } catch (const std::exception& e) {
             *     std::cerr << e.what() << std::endl;
             * }
             * ```
             * @endcode
             */
            explicit InputNode(const Tensor::shape_type& shape, const std::initializer_list<Tensor::value_type>& data,
                               bool requires_grad = false);

            /**
             * @brief Forward pass for the `InputNode`.
             *
             * The `forward()` method for the `InputNode` is a no-op (no operation) because the input node does not perform any
             * computations during the forward pass. Its primary role is to provide the input data to the computational graph.
             *
             * Since the `InputNode` does not modify its data or perform any calculations, the `forward()` method does not
             * need to be implemented with any functionality, and it is left empty. The tensor stored in the `output` member
             * will simply be passed along to the next nodes in the computational graph during the forward pass.
             *
             * @note
             * - This method is a required implementation due to inheritance from the abstract `Node` class, but it does not perform
             *   any operations for `InputNode`. It serves as a placeholder to conform to the `Node` class interface.
             * - The `InputNode` only holds input data, and its `forward()` method ensures that the data is available for the
             *   subsequent nodes in the graph.
             *
             * @see backward() for the reverse propagation (gradient calculation) method.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void forward() override;

            /**
             * @brief Backward pass for the `InputNode`.
             *
             * The `backward()` method for the `InputNode` is a no-op (no operation) because the input node does not participate
             * in the backpropagation process. It does not have any parameters or gradients to update, as its role is simply to provide
             * the input data to the computational graph.
             *
             * Since the `InputNode` does not modify any data during the backward pass, the `backward()` method does not need to
             * perform any operations, and it is left empty. Gradients will not be propagated from this node to any other nodes,
             * as the `InputNode` does not require gradients.
             *
             * @note
             * - This method is a required implementation due to inheritance from the abstract `Node` class, but it does not perform
             *   any operations for `InputNode`. It serves as a placeholder to conform to the `Node` class interface.
             * - The `InputNode` is used to provide data to the computational graph, but since it doesn't have parameters, there is no
             *   need to propagate gradients through it.
             *
             * @see forward() for the forward pass computation method.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void backward() override;
        };

        /**
         * @class OutputNode
         * @brief Base class for loss function nodes in a computational graph.
         *
         * The `OutputNode` class serves as the base class for all nodes representing loss functions in a neural network.
         * It connects to the output of a node that produces the final result, and it computes the loss based on that result.
         * During the forward pass, it simply copies the output of the input node, and during the backward pass, it sets
         * the gradient of the output tensor to 1, effectively marking the end of the gradient flow.
         *
         * The `OutputNode` class is used as a parent class for more specific loss function nodes (such as Mean Squared Error or
         * Cross-Entropy loss), which can further extend its functionality to compute the actual loss and update the `loss` member.
         *
         * @details
         * Key features:
         * - **Loss Calculation**: The `loss` member variable holds the value of the computed loss. Specific loss functions
         *   can update this value by extending the `OutputNode` class.
         * - **Forward Pass**: The `forward()` method simply sets the `output` member to the output of the input node.
         * - **Backward Pass**: The `backward()` method sets the gradient of the output tensor to 1, which marks the start of
         *   gradient propagation for the backward pass.
         * - **Loss Access**: The `getLoss()` method provides access to the loss value stored in the `loss` member.
         *
         * This class is part of the `nz::nodes` namespace, and it is designed to be extended for implementing various
         * loss functions.
         *
         * @note
         * - The `OutputNode` class does not perform any specific loss computation. It is intended to be a base class for
         *   more specific loss function nodes that compute and track the actual loss.
         * - The backward pass simply sets the gradient to 1, which is appropriate for the output layer of a neural network where
         *   the loss gradient is propagated back.
         *
         * ### Usage Example:
         * ```cpp
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3} and requires gradients
         * input.output->fill(1.0f);  // Fill the input tensor with 1.0
         *
         * OutputNode output(&input);  // Create an OutputNode and pass the input node as the source
         * output.forward();  // Forward pass: output now points to the input node's output
         * output.backward();  // Backward pass: set the gradient of the input node's output to 1
         *
         * std::cout << "Loss: " << output.getLoss() << std::endl;  // Access the loss value (which is 0 initially)
         * ```
         *
         * @see forward() for the forward pass computation method.
         * @see backward() for the backward pass gradient propagation method.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/11/29
         */
        class DL_API OutputNode : public Node {
        protected:
            Tensor::value_type loss;

        public:
            /**
             * @brief Constructor to initialize an `OutputNode` with a given input node.
             *
             * This constructor initializes an `OutputNode` by accepting an input node. The `output` of this node will be
             * set to the `output` of the provided input node during the forward pass. The `loss` is initialized to `0`, and
             * the `type` is set to `"Output"`.
             *
             * The `OutputNode` class is designed to represent the output layer of a neural network, and it serves as the base
             * class for loss function nodes. The `forward()` and `backward()` methods will be responsible for propagating
             * data and gradients, respectively.
             *
             * @param input A pointer to the `Node` that serves as the input to the `OutputNode`. The `output` of this node
             *              will be used as the `OutputNode`'s output.
             *
             * This constructor sets up the node with a reference to its input, allowing the `OutputNode` to pass data from
             * its input node and compute the loss during the forward and backward passes.
             *
             * @note
             * - The `InputNode` or any other node that provides the final output of the network can be passed to this constructor.
             * - The `loss` member is initialized to `0` and can be updated by specific loss function implementations in derived classes.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            explicit OutputNode(Node* input);

            /**
             * @brief Forward pass for the `OutputNode`.
             *
             * The `forward()` method for the `OutputNode` sets the `output` member of the node to be the same as the `output`
             * of its input node. This effectively passes the output from the input node to the `OutputNode` without any modification.
             * Since the `OutputNode` does not perform any computation itself, it simply relays the input node's output during the
             * forward pass, making it equivalent to its input node's output.
             *
             * This method is typically used in the context of a neural network, where the `OutputNode` represents the final layer,
             * and it connects the output of the network to the loss function for loss computation and backpropagation.
             *
             * @note
             * - The `forward()` method does not alter the data from the input node; it merely sets the `output` of the `OutputNode`
             *   to be the same as the input node's `output`.
             * - This method is implemented as part of the `OutputNode` class to conform to the interface defined by its base class `Node`.
             *
             * @see backward() for the backward pass gradient propagation method.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void forward() override;

            /**
             * @brief Backward pass for the `OutputNode`.
             *
             * The `backward()` method for the `OutputNode` sets the gradient of the output tensor to 1. If the input tensor
             * of the `OutputNode` requires gradients (i.e., it is part of the model parameters), the gradient of the input tensor
             * is set to 1. This is a standard operation in the backward pass for the output layer, as it marks the start of the
             * gradient propagation in the network.
             *
             * This method does not perform any gradient calculations for the output node itself. Instead, it ensures that the
             * gradient of the input node’s output is set to 1, which is necessary for the backpropagation process in the neural network.
             *
             * @note
             * - The `backward()` method simply fills the gradient of the input tensor with 1. This is because the `OutputNode` represents
             *   the output layer, where the gradient is typically set to 1 as the starting point of backpropagation.
             * - This method ensures that the gradient for the input node’s `output` is available for further propagation through
             *   the network during the backward pass.
             *
             * @see forward() for the forward pass computation method.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void backward() override;

            /**
             * @brief Retrieves the loss value stored in the `OutputNode`.
             *
             * The `getLoss()` method returns the value of the loss that is stored in the `loss` member of the `OutputNode`.
             * This value is typically updated by a derived class (e.g., a specific loss function class like Mean Squared Error or
             * Cross-Entropy Loss) during the forward pass. The `loss` represents the discrepancy between the predicted output and
             * the actual target output in the context of a neural network.
             *
             * The `getLoss()` function provides access to the computed loss value, which is essential for monitoring the
             * network’s performance during training and optimization.
             *
             * @return The current loss value stored in the `loss` member, which is of type `Tensor::value_type`.
             *
             * @note
             * - This method does not modify the loss value; it simply returns the current stored loss value.
             * - The actual loss computation happens in the derived class, such as `MeanSquaredErrorNode` or `BinaryCrossEntropyNode`.
             *
             * @see forward() for the forward pass computation method where the loss might be updated.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            [[nodiscard]] Tensor::value_type getLoss() const;

            /**
             * @brief Prints the type, data, gradient, and loss of the node.
             *
             * The `print()` method outputs the information about the node, including its type, the tensor data stored in the node's
             * output, the corresponding gradient, and the loss value (if available). This is useful for debugging and inspecting
             * the state of nodes in a computational graph or during training, allowing for easy visualization of the node's content,
             * gradients, and any associated loss.
             *
             * The method outputs the following details:
             * - **Type**: The type of the node (e.g., the operation it represents, such as "MatrixMul", "ReLU", etc.).
             * - **Data**: The tensor data stored in the node's `output` tensor.
             * - **Gradient**: If the node has a computed gradient, it is also displayed, providing insights into the gradient values
             *   that are being backpropagated through the network during training.
             * - **Loss**: The loss value associated with the node (if applicable). This value can be used to track the error or
             *   discrepancy during the forward-backward pass in training.
             *
             * This method is primarily used for debugging and monitoring the state of tensors, gradients, and loss, making it easier
             * to inspect how the data, gradients, and error values flow through the network.
             *
             * @note
             * - The `output` tensor should contain both the data and the gradient information, and both are printed when this
             *   method is called.
             * - The `loss` value will only be printed if it is associated with the node. If the node does not have a loss value,
             *   this field may be omitted.
             * - This method is typically used during development or debugging phases and should not be used in performance-critical
             *   code as it involves printing potentially large amounts of data.
             *
             * @param os The output stream (e.g., `std::cout`) to which the node's information will be printed.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void print(std::ostream& os) const override;
        };
    }

    /**
     * @namespace nz::nodes::calc
     * @brief Contains classes and functionality for computation nodes in a neural network or computational graph.
     *
     * The `nz::nodes::calc` namespace provides a collection of classes that represent various
     * computational operations within a neural network. These nodes perform essential mathematical operations
     * during the forward pass in a computational graph.
     *
     * This namespace includes:
     * - **Computation Nodes**: Nodes responsible for performing mathematical operations and data transformations
     *   such as activation functions, matrix operations, and normalization techniques.
     *   - **Activation Functions**: Nodes such as `ReLUNode`, `SigmoidNode`, `TanhNode`, `LeakyReLUNode`, etc., which
     *     apply non-linear transformations to the input data.
     *   - **Mathematical Operations**: Nodes like `AddNode`, `MatMulNode`, `ScalarMulNode`, `ScalarDivNode`, and others
     *     for performing basic arithmetic and matrix operations.
     *
     * The nodes in this namespace interact with `Tensor` objects, performing data manipulation operations necessary
     * for building neural network layers and facilitating the forward propagation of information through the network.
     *
     * @note
     * - The nodes in this namespace are intended for mathematical and computational operations in the forward pass.
     * - These nodes can be combined with other nodes to form complex neural network architectures.
     * - Make sure to manage memory properly when working with large tensors, particularly when leveraging GPU resources.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/11/29
     */
    namespace calc {
        /**
         * @class AddNode
         * @brief Represents a node that performs element-wise addition between two input tensors.
         *
         * The `AddNode` class is a computational node that performs element-wise addition between two input tensors
         * during the forward pass. It also handles the backpropagation of gradients during the backward pass, propagating
         * the gradient of the output tensor back to both input tensors. This node is typically used to represent addition
         * operations in neural network computations.
         *
         * @details
         * Key features:
         * - **Forward Pass**: The `forward()` method performs element-wise addition of the two input tensors and stores
         *   the result in the `output` tensor.
         * - **Backward Pass**: The `backward()` method propagates the gradient of the output tensor to both input tensors
         *   by copying the gradient of the output to the gradients of the inputs.
         * - **Shape Check**: The constructor checks that the shapes of the two input tensors are the same, as element-wise
         *   addition requires matching shapes.
         *
         * This class is part of the `nz::nodes` namespace and is designed for use in a computational graph where
         * addition operations are needed.
         *
         * @note
         * - The `AddNode` is specifically for element-wise addition. The shapes of the input tensors must match.
         * - During the backward pass, the gradient of the output is distributed equally to both inputs.
         *
         * ### Usage Example:
         * ```cpp
         * // Example 1: Creating and using an AddNode
         * InputNode input1({3, 3}, true);  // Create the first input node with shape {3, 3}
         * input1.output->fill(1.0f);  // Fill the tensor with value 1.0
         *
         * InputNode input2({3, 3}, true);  // Create the second input node with shape {3, 3}
         * input2.output->fill(2.0f);  // Fill the tensor with value 2.0
         *
         * AddNode add_node(&input1, &input2);  // Create an AddNode using the two input nodes
         * add_node.forward();  // Perform the forward pass: output = input1 + input2
         * add_node.backward();  // Perform the backward pass: propagate gradients
         *
         * std::cout << "Output: " << *add_node.output << std::endl;  // Print the output tensor
         * ```
         *
         * @see forward() for the forward pass computation method.
         * @see backward() for the backward pass gradient propagation method.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/11/29
         */
        class DL_API AddNode : public Node {
        public:
            /**
             * @brief Constructor to initialize an `AddNode` with two input nodes for element-wise addition.
             *
             * The constructor initializes an `AddNode` that performs element-wise addition between the outputs of two input nodes.
             * It ensures that the shapes of the two input tensors are compatible for the addition operation. If the shapes of
             * the two input tensors do not match, an exception is thrown. The constructor also sets up the output tensor and
             * determines whether gradients need to be tracked based on the inputs' `requiresGrad` property.
             *
             * @param input_left A pointer to the first input node. Its `output` tensor is used in the addition operation.
             * @param input_right A pointer to the second input node. Its `output` tensor is used in the addition operation.
             *
             * The constructor verifies that the two input tensors have the same shape, and initializes the `output` tensor
             * with the same shape as the inputs. The `requires_grad` flag for the output tensor is set to `true` if either
             * of the input tensors requires gradients.
             *
             * @throws std::invalid_argument If the shapes of the input tensors do not match.
             *
             * @note
             * - The constructor checks for shape compatibility between the two input tensors, and if the shapes do not match,
             *   it throws an exception to prevent invalid operations.
             * - The `output` tensor is created with the same shape as the input tensors, and will track gradients if any of
             *   the input tensors require them.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            AddNode(Node* input_left, Node* input_right);

            /**
             * @brief Forward pass for the `AddNode` to perform element-wise addition.
             *
             * The `forward()` method performs the element-wise addition between the two input tensors and stores the result
             * in the `output` tensor. It uses CUDA kernel `MatrixAddKernel` to carry out the addition operation efficiently on the GPU.
             *
             * This method is called during the forward pass of the neural network, where it computes the sum of the two input tensors
             * and assigns the result to the `output` tensor. The shape of the `output` tensor will be the same as the shape of the
             * input tensors, as verified during the initialization of the `AddNode`.
             *
             * The method divides the work into blocks and grids to parallelize the addition operation over the GPU.
             *
             * @note
             * - The `MatrixAdd` kernel performs the addition operation on the GPU, ensuring efficient parallel computation.
             * - The `output` tensor is updated with the result of the addition, and it must be allocated before calling `forward()`.
             *
             * @see backward() for the backward pass gradient propagation method.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void forward() override;

            /**
             * @brief Backward pass for the `AddNode` to propagate gradients.
             *
             * The `backward()` method propagates the gradient of the `output` tensor to the gradients of the two input tensors
             * during the backward pass. Since addition is an element-wise operation, the gradient of the output is propagated
             * equally to both input tensors.
             *
             * If either of the input tensors requires gradients (i.e., its `requiresGrad()` method returns `true`), the gradient
             * of the output tensor (`output->grad()`) is copied to the gradient of the corresponding input tensor. This is done
             * using `cudaMemcpy` to efficiently propagate the gradients on the GPU.
             *
             * This method is typically called during the backpropagation step of neural network training, where gradients are
             * propagated backward through the network, starting from the output layer.
             *
             * @note
             * - The gradient of the output tensor is propagated to both input tensors, and each input receives the exact same
             *   gradient as the output.
             * - This method does not compute gradients for the output tensor itself. It simply propagates the gradient from the
             *   output to the inputs.
             * - The gradients are copied using `cudaMemcpy` to ensure efficient GPU-based gradient propagation.
             *
             * @see forward() for the forward pass computation method.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void backward() override;
        };

        /**
         * @class MatMulNode
         * @brief Represents a matrix multiplication operation node in a computational graph.
         *
         * The `MatMulNode` class performs matrix multiplication between two input tensors. It implements the matrix
         * multiplication operation in the forward pass, and propagates the gradients during the backward pass.
         * This node is typically used to represent fully connected layers or other linear algebraic operations
         * in a neural network or computational graph. The node now leverages Tensor Cores for efficient half-precision
         * matrix multiplication, improving performance during forward and backward passes.
         *
         * @details
         * Key features:
         * - **Forward Pass**: The `forward()` method computes the matrix multiplication of two input tensors and stores
         *   the result in the `output` tensor. The computation is accelerated using Tensor Cores with half-precision (FP16)
         *   to speed up matrix multiplication operations.
         * - **Backward Pass**: The `backward()` method propagates the gradients from the output tensor to the input tensors
         *   using the chain rule of calculus.
         * - **Shape Check**: The constructor ensures that the number of columns in the left input tensor matches the number
         *   of rows in the right input tensor, as required for matrix multiplication.
         *
         * This class is part of the `nz::nodes` namespace and is used for matrix operations in a computational graph.
         *
         * @note
         * - The left input tensor's number of columns must match the right input tensor's number of rows.
         * - The matrix multiplication operation in this node uses Tensor Cores for faster computation using half-precision
         *   floating-point arithmetic (FP16).
         *
         * ### Usage Example:
         * ```cpp
         * // Example 1: Creating and using a MatMulNode
         * InputNode input1({3, 2}, true);  // Create the first input node with shape {3, 2}
         * input1.output->fill(1.0f);  // Fill the tensor with value 1.0
         *
         * InputNode input2({2, 3}, true);  // Create the second input node with shape {2, 3}
         * input2.output->fill(2.0f);  // Fill the tensor with value 2.0
         *
         * MatMulNode matmul_node(&input1, &input2);  // Create a MatMulNode using the two input nodes
         * matmul_node.forward();  // Perform the forward pass: output = input1 * input2
         * matmul_node.backward();  // Perform the backward pass: propagate gradients
         *
         * std::cout << "Output: " << *matmul_node.output << std::endl;  // Print the output tensor
         * ```
         *
         * @see forward() for the forward pass computation method.
         * @see backward() for the backward pass gradient propagation method.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/11/29
         */
        class DL_API MatMulNode : public Node {
        public:
            /**
             * @brief Constructor to initialize a `MatMulNode` for matrix multiplication.
             *
             * This constructor initializes an `MatMulNode` which performs matrix multiplication between the outputs
             * of two input nodes. It ensures that the shapes of the two input tensors are compatible for matrix multiplication.
             * Specifically, the number of columns of the left input tensor must match the number of rows of the right input tensor.
             * If the shapes do not match, an exception is thrown. The constructor also initializes the `output` tensor with
             * the appropriate shape based on the input tensors and sets the `requires_grad` flag based on the input tensors'
             * gradient tracking requirements.
             *
             * @param input_left A pointer to the first input node. Its `output` tensor is used for the matrix multiplication.
             * @param input_right A pointer to the second input node. Its `output` tensor is used for the matrix multiplication.
             *
             * The constructor checks that the number of columns in the left input tensor (`input_left->output->shape()[1]`)
             * matches the number of rows in the right input tensor (`input_right->output->shape()[0]`), as required for matrix
             * multiplication. The output tensor is created with the shape `(input_left->output->shape()[0], input_right->output->shape()[1])`,
             * and the `requires_grad` flag is set to `true` if either of the input tensors requires gradients.
             *
             * @throws std::invalid_argument If the shapes of the input tensors are not compatible for matrix multiplication.
             *
             * @note
             * - The left input tensor's column count must match the right input tensor's row count for matrix multiplication.
             * - The constructor ensures that the output tensor has the correct shape to hold the result of the matrix multiplication.
             * - The `requires_grad` flag for the output tensor is set based on the gradient requirements of the input tensors.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            MatMulNode(Node* input_left, Node* input_right);

            /**
             * @brief Forward pass for the `MatMulNode` to perform matrix multiplication.
             *
             * The `forward()` method computes the matrix multiplication between the two input tensors using CUDA, and stores
             * the result in the `output` tensor. The matrix multiplication is performed using the `GeneralMatrixMul` kernel
             * on the GPU, which efficiently computes the product of the two matrices in parallel.
             *
             * This method is called during the forward pass of the neural network. It calculates the matrix product of the left
             * input tensor (`inputs[0]`) and the right input tensor (`inputs[1]`), and stores the result in the `output` tensor.
             * The shape of the `output` tensor is determined by the number of rows in the left input tensor and the number of
             * columns in the right input tensor.
             *
             * @note
             * - The kernel `GeneralMatrixMul` performs the matrix multiplication using parallel computation on the GPU.
             * - The matrix multiplication is performed as `M = A * B`, where `A` is the left input tensor and `B` is the right input tensor.
             * - The block size (`TILE_SIZE`) and grid size are chosen to ensure efficient GPU parallelization of the operation.
             *
             * @see backward() for the backward pass gradient propagation method.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void forward() override;

            /**
             * @brief Backward pass for the `MatMulNode` to propagate gradients.
             *
             * The `backward()` method computes the gradients of the input tensors with respect to the output tensor for the
             * matrix multiplication operation. During the backward pass, the gradients of the output tensor are propagated
             * back to the two input tensors. The gradient computation follows the chain rule of calculus.
             *
             * Specifically:
             * - For the left input tensor (`A`), the gradient is computed as `dA = dC * B^T`, where `dC` is the gradient
             *   of the output tensor and `B^T` is the transpose of the right input tensor.
             * - For the right input tensor (`B`), the gradient is computed as `dB = A^T * dC`, where `A^T` is the transpose
             *   of the left input tensor and `dC` is the gradient of the output tensor.
             *
             * These gradients are computed on the GPU using CUDA kernels (`GeneralMatrixMul`), which parallelize the
             * matrix operations.
             *
             * @note
             * - The gradients for both input tensors are computed only if they require gradients (i.e., `requiresGrad()` is true).
             * - The gradients are computed using the transposes of the input matrices and propagated through the network.
             * - The `GeneralMatrixMul` kernel is used for efficient gradient computation on the GPU.
             *
             * @see forward() for the forward pass computation method.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date
             * 2024/11/29
             */
            void backward() override;
        };

        /**
         * @class ScalarMulNode
         * @brief Represents a scalar multiplication operation node in a computational graph.
         *
         * The `ScalarMulNode` class performs element-wise multiplication of a tensor by a scalar value.
         * It is used in computational graphs to implement scalar scaling of tensors, which is common in
         * various machine learning and numerical computing tasks.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Multiplies the input tensor's elements by a scalar value and stores the result
         *   in the `output` tensor.
         * - **Backward Pass**: Propagates gradients from the `output` tensor back to the input tensor by
         *   scaling the gradient with the scalar value.
         * - **Scalar Handling**: Handles scalar operations directly but issues a warning that scalar operations
         *   do not support saving to files, encouraging the use of matrix operations when persistence is required.
         *
         * This class is part of the `nz::nodes` namespace and is used for scalar-tensor operations
         * in a computational graph.
         *
         * @note
         * - This node is designed for scalar-tensor operations. It does not support saving scalar operations
         *   to files; users should rely on matrix operations for this purpose.
         * - The scalar value is stored internally and used for both the forward and backward passes.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using ScalarMulNode for scalar multiplication
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         * input.output->fill(2.0f);  // Fill the input tensor with value 2.0
         *
         * ScalarMulNode scalar_mul_node(&input, 5.0f);  // Multiply the input tensor by 5.0
         * scalar_mul_node.forward();  // Perform the forward pass
         * scalar_mul_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *scalar_mul_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the forward pass computation method.
         * @see backward() for the backward pass gradient propagation method.
         *
         * @warning
         * - Scalar operations do not yet support saving to files.
         * - Use matrix operations instead if model saving is required.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/05
         */
        class DL_API ScalarMulNode : public Node {
            Tensor::value_type scalar;

        public:
            /**
             * @brief Constructor to initialize a `ScalarMulNode` for scalar multiplication.
             *
             * The constructor initializes a `ScalarMulNode`, which performs element-wise multiplication of the
             * output tensor of the input node by a scalar value. It sets up the node's input connections,
             * determines the gradient tracking requirement, and prepares the output tensor with the appropriate
             * shape and properties.
             *
             * @param input A pointer to the input node. Its `output` tensor will be multiplied by the scalar value.
             * @param scalar The scalar value to multiply with the input tensor.
             *
             * @details
             * - The constructor connects the input node to this `ScalarMulNode` by adding the input node to
             *   the `inputs` vector.
             * - It initializes the `output` tensor with the same shape as the input tensor and determines
             *   whether the output tensor should track gradients based on the input tensor's gradient requirement.
             * - A warning is issued to inform the user that scalar operations currently do not support saving
             *   to files, and matrix operations should be used for models requiring persistence.
             *
             * @note
             * - The scalar value is stored internally in the `ScalarMulNode` instance and used during the
             *   forward and backward passes.
             *
             * @warning
             * - Scalar operations are not yet supported for saving to files. Use matrix operations as an alternative.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for the gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            ScalarMulNode(Node* input, Tensor::value_type scalar);

            /**
             * @brief Forward pass for the `ScalarMulNode` to perform scalar multiplication.
             *
             * The `forward()` method computes the element-wise multiplication of the input tensor by the scalar value.
             * It uses CUDA kernels to perform the computation in parallel on the GPU, ensuring efficient execution.
             *
             * @details
             * - The method launches a CUDA kernel (`ScalarMul`) to compute the scalar multiplication.
             * - The output tensor is populated with the results of the multiplication.
             * - The grid and block dimensions are calculated based on the size of the output tensor to optimize GPU performance.
             *
             * @note
             * - The scalar multiplication is performed as `output[i] = input[i] * scalar` for each element in the tensor.
             *
             * @see backward() for gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `ScalarMulNode` to propagate gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor by
             * scaling the gradient of the output tensor using the scalar value. This operation ensures that the
             * gradients are correctly propagated back through the computational graph.
             *
             * @details
             * - The method checks if the input tensor requires gradients. If not, no computation is performed.
             * - A CUDA kernel (`ScalarMul`) is launched to scale the gradient of the output tensor and accumulate
             *   the result in the gradient of the input tensor.
             * - The grid and block dimensions are calculated based on the size of the output tensor to ensure
             *   efficient GPU parallelization.
             *
             * @note
             * - The gradient computation is performed as `grad_input[i] = grad_output[i] * scalar` for each element
             *   in the tensor, where `grad_output` is the gradient of the `output` tensor.
             *
             * @see forward() for the scalar multiplication computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class ScalarDivNode
         * @brief Represents a scalar division operation node in a computational graph.
         *
         * The `ScalarDivNode` class performs element-wise division of a tensor by a scalar value.
         * It is used in computational graphs to implement normalization or scaling operations, which
         * are fundamental in machine learning and numerical computations.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Divides each element of the input tensor by a scalar value and stores the
         *   result in the `output` tensor.
         * - **Backward Pass**: Propagates gradients from the `output` tensor back to the input tensor by
         *   scaling the gradients with the reciprocal of the scalar value.
         * - **Error Handling**: Ensures that the scalar value is non-zero during construction to prevent
         *   division by zero errors.
         * - **Shape Preservation**: Maintains the shape of the input tensor in the `output` tensor.
         * - **Gradient Management**: Tracks whether gradients are required for the operation based on the
         *   properties of the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is optimized for scalar-tensor division
         * operations in computational graphs.
         *
         * @note
         * - The scalar value must be non-zero. An exception is thrown during construction if this condition is not met.
         * - A warning is issued indicating that scalar operations do not support saving to files, and users are encouraged
         *   to use matrix operations for model persistence.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using ScalarDivNode for scalar division
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         * input.output->fill(10.0f);  // Fill the input tensor with value 10.0
         *
         * ScalarDivNode scalar_div_node(&input, 2.0f);  // Divide the input tensor by 2.0
         * scalar_div_node.forward();  // Perform the forward pass
         * scalar_div_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *scalar_div_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the forward pass implementation.
         * @see backward() for the gradient propagation in the backward pass.
         *
         * @throws std::invalid_argument If the scalar value is zero, as division by zero is undefined.
         *
         * @warning
         * - Scalar operations are not yet supported for saving to files. Use matrix operations as an alternative.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API ScalarDivNode : public Node {
            Tensor::value_type scalar;

        public:
            /**
             * @brief Constructor to initialize a `ScalarDivNode` for scalar division.
             *
             * The constructor initializes a `ScalarDivNode`, which performs element-wise division of the
             * output tensor of the input node by a scalar value. It validates the scalar value, sets up the node's
             * input connections, determines the gradient tracking requirement, and prepares the output tensor
             * with the appropriate shape and properties.
             *
             * @param input A pointer to the input node. Its `output` tensor will be divided by the scalar value.
             * @param scalar The scalar value to divide the input tensor by.
             *
             * @details
             * - The constructor verifies that the scalar value is non-zero. If the scalar is zero, an exception
             *   is thrown to prevent division by zero errors.
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - It initializes the `output` tensor with the same shape as the input tensor and determines whether
             *   the output tensor should track gradients based on the input tensor's gradient requirement.
             * - A warning is issued to inform the user that scalar operations currently do not support saving to files,
             *   encouraging the use of matrix operations for models requiring persistence.
             *
             * @note
             * - The scalar value must be non-zero to ensure valid division. An exception will be thrown if this
             *   condition is not met.
             * - The scalar value is stored internally in the `ScalarDivNode` instance and used during the forward
             *   and backward passes.
             *
             * @throws std::invalid_argument If the scalar value is zero, as division by zero is undefined.
             *
             * @warning
             * - Scalar operations are not yet supported for saving to files. Use matrix operations as an alternative.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for the gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            ScalarDivNode(Node* input, Tensor::value_type scalar);

            /**
             * @brief Forward pass for the `ScalarDivNode` to perform scalar division.
             *
             * The `forward()` method computes the element-wise division of the input tensor by the scalar value.
             * It uses CUDA kernels to execute the operation efficiently on the GPU.
             *
             * @details
             * - A CUDA kernel (`ScalarDiv`) is launched to compute the division for each element in the input tensor.
             * - The grid and block dimensions are calculated dynamically based on the size of the output tensor
             *   to optimize parallel computation on the GPU.
             * - The result of the division is stored in the `output` tensor.
             *
             * @note
             * - The division operation is performed as `output[i] = input[i] / scalar` for each element of the tensor.
             * - Ensure the scalar value is non-zero, as division by zero will result in undefined behavior.
             *
             * @see backward() for gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `ScalarDivNode` to propagate gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor by
             * scaling the gradient of the output tensor using the reciprocal of the scalar value. This ensures
             * the gradients are correctly propagated back through the computational graph.
             *
             * @details
             * - The method first checks if the input tensor requires gradients. If true, a CUDA kernel (`ScalarDiv`)
             *   is launched to compute the scaled gradients.
             * - The gradient computation is performed as `grad_input[i] = grad_output[i] / scalar` for each element,
             *   where `grad_output` is the gradient of the `output` tensor.
             * - The resulting gradients are stored in the gradient tensor of the input node.
             *
             * @note
             * - The backward pass uses the same scalar value as in the forward pass, ensuring consistency in
             *   gradient computation.
             * - The scalar value must be non-zero to avoid undefined behavior.
             *
             * @see forward() for the scalar division computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class ScalarAddNode
         * @brief Represents a scalar addition operation node in a computational graph.
         *
         * The `ScalarAddNode` class performs element-wise addition of a tensor and a scalar value.
         * It is commonly used in computational graphs for offsetting tensor values or applying
         * a bias term in various operations.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Adds a scalar value to each element of the input tensor and stores the
         *   result in the `output` tensor.
         * - **Backward Pass**: Propagates gradients from the `output` tensor back to the input tensor
         *   without modification, as the derivative of addition with respect to the input is 1.
         * - **Shape Preservation**: Maintains the shape of the input tensor in the `output` tensor.
         * - **Gradient Management**: Tracks whether gradients are required for the operation based on the
         *   properties of the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and facilitates scalar-tensor addition
         * operations in computational graphs.
         *
         * @note
         * - The scalar value is applied consistently across all elements of the input tensor.
         * - A warning is issued indicating that scalar operations do not support saving to files, and users
         *   are encouraged to use matrix operations for model persistence.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using ScalarAddNode for scalar addition
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         * input.output->fill(10.0f);  // Fill the input tensor with value 10.0
         *
         * ScalarAddNode scalar_add_node(&input, 5.0f);  // Add 5.0 to the input tensor
         * scalar_add_node.forward();  // Perform the forward pass
         * scalar_add_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *scalar_add_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the scalar addition computation in the forward pass.
         * @see backward() for gradient propagation in the backward pass.
         *
         * @warning
         * - Scalar operations are not yet supported for saving to files. Use matrix operations as an alternative.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API ScalarAddNode : public Node {
            Tensor::value_type scalar;

        public:
            /**
             * @brief Constructor to initialize a `ScalarAddNode` for scalar addition.
             *
             * The constructor initializes a `ScalarAddNode`, which performs element-wise addition of a tensor
             * and a scalar value. It establishes the connection between the input node and this node, prepares
             * the output tensor with the appropriate shape and properties, and stores the scalar value for use
             * during forward and backward passes.
             *
             * @param input A pointer to the input node. Its `output` tensor will be added to the scalar value.
             * @param scalar The scalar value to add to each element of the input tensor.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and the `requires_grad`
             *   property is determined based on the input tensor's gradient requirements.
             * - A warning is issued indicating that scalar operations do not support saving to files, encouraging
             *   the use of matrix operations for models requiring persistence.
             *
             * @note
             * - The scalar value is applied consistently across all elements of the input tensor during the forward pass.
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient propagation in the backward pass.
             *
             * @warning
             * - Scalar operations are not yet supported for saving to files. Use matrix operations as an alternative.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            ScalarAddNode(Node* input, Tensor::value_type scalar);

            /**
             * @brief Forward pass for the `ScalarAddNode` to perform scalar addition.
             *
             * The `forward()` method computes the element-wise addition of the input tensor and the scalar value.
             * It uses a CUDA kernel to efficiently execute the operation in parallel on the GPU.
             *
             * @details
             * - A CUDA kernel (`ScalarAdd`) is launched to add the scalar value to each element of the input tensor.
             * - The grid and block dimensions are dynamically calculated based on the size of the output tensor
             *   to optimize GPU parallelism.
             * - The result of the addition is stored in the `output` tensor.
             *
             * @note
             * - The addition operation is performed as `output[i] = input[i] + scalar` for each element of the tensor.
             * - The scalar value is applied uniformly to all elements in the input tensor.
             *
             * @see backward() for gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `ScalarAddNode` to propagate gradients.
             *
             * The `backward()` method propagates the gradient of the loss from the output tensor back to the input tensor.
             * Since the derivative of addition with respect to the input is 1, the gradient from the output tensor is
             * directly copied to the input tensor's gradient.
             *
             * @details
             * - The method first checks if the input tensor requires gradients. If true, the gradient of the `output` tensor
             *   is copied to the gradient of the input tensor using `cudaMemcpy`.
             * - The operation ensures efficient gradient propagation without any additional computation.
             *
             * @note
             * - The backward pass assumes that the gradient of the `output` tensor is already computed and properly initialized.
             * - This method does not modify the gradient values; it performs a direct transfer.
             *
             * @see forward() for the scalar addition computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class ScalarSubNode
         * @brief Represents a scalar subtraction operation node in a computational graph.
         *
         * The `ScalarSubNode` class performs element-wise subtraction of a scalar value from a tensor.
         * It is commonly used in computational graphs to offset tensor values or perform subtraction-based
         * normalization tasks.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Subtracts a scalar value from each element of the input tensor and stores the
         *   result in the `output` tensor.
         * - **Backward Pass**: Propagates gradients from the `output` tensor back to the input tensor. Since
         *   the derivative of subtraction with respect to the input is 1, the gradient from the `output` tensor
         *   is directly transferred to the input tensor.
         * - **Shape Preservation**: Maintains the shape of the input tensor in the `output` tensor.
         * - **Gradient Management**: Tracks whether gradients are required for the operation based on the
         *   properties of the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and facilitates scalar-tensor subtraction
         * operations in computational graphs.
         *
         * @note
         * - The scalar value is applied consistently across all elements of the input tensor.
         * - A warning is issued indicating that scalar operations do not support saving to files, and users
         *   are encouraged to use matrix operations for model persistence.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using ScalarSubNode for scalar subtraction
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         * input.output->fill(10.0f);  // Fill the input tensor with value 10.0
         *
         * ScalarSubNode scalar_sub_node(&input, 5.0f);  // Subtract 5.0 from the input tensor
         * scalar_sub_node.forward();  // Perform the forward pass
         * scalar_sub_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *scalar_sub_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the scalar subtraction computation in the forward pass.
         * @see backward() for gradient propagation in the backward pass.
         *
         * @warning
         * - Scalar operations are not yet supported for saving to files. Use matrix operations as an alternative.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API ScalarSubNode : public Node {
            Tensor::value_type scalar;

        public:
            /**
             * @brief Constructor to initialize a `ScalarSubNode` for scalar subtraction.
             *
             * The constructor initializes a `ScalarSubNode`, which performs element-wise subtraction of a scalar
             * value from the elements of the input tensor. It establishes the connection between the input node
             * and this node, prepares the output tensor with the appropriate shape and properties, and stores the
             * negated scalar value for use during forward and backward passes.
             *
             * @param input A pointer to the input node. Its `output` tensor will have the scalar value subtracted from it.
             * @param scalar The scalar value to subtract from each element of the input tensor.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and the `requires_grad`
             *   property is determined based on the input tensor's gradient requirements.
             * - The scalar value is negated and stored internally for efficient computation during the forward pass.
             * - A warning is issued indicating that scalar operations do not support saving to files, encouraging
             *   the use of matrix operations for models requiring persistence.
             *
             * @note
             * - The negation of the scalar value simplifies computation during the forward pass, treating subtraction
             *   as addition with a negated scalar.
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient propagation in the backward pass.
             *
             * @warning
             * - Scalar operations are not yet supported for saving to files. Use matrix operations as an alternative.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            ScalarSubNode(Node* input, Tensor::value_type scalar);

            /**
             * @brief Forward pass for the `ScalarSubNode` to perform scalar subtraction.
             *
             * The `forward()` method computes the element-wise subtraction of a scalar value from the input tensor.
             * Internally, it utilizes the addition kernel (`ScalarAdd`) by treating the subtraction as addition with
             * a negated scalar value, which was preprocessed during node construction.
             *
             * @details
             * - A CUDA kernel (`ScalarAdd`) is launched to add the negated scalar value to each element of the input tensor.
             * - The grid and block dimensions are dynamically calculated based on the size of the output tensor to
             *   optimize GPU parallelism.
             * - The result of the operation is stored in the `output` tensor.
             *
             * @note
             * - The subtraction operation is effectively performed as `output[i] = input[i] - scalar`, achieved by
             *   using `output[i] = input[i] + (-scalar)` for efficiency.
             * - The scalar value was negated during construction, making this method consistent with the addition kernel.
             *
             * @see backward() for gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `ScalarSubNode` to propagate gradients.
             *
             * The `backward()` method propagates the gradient of the loss from the output tensor back to the input tensor.
             * Since the derivative of subtraction with respect to the input is 1, the gradient from the output tensor
             * is directly copied to the input tensor's gradient.
             *
             * @details
             * - The method checks if the input tensor requires gradients. If true, the gradient of the `output` tensor
             *   is copied directly to the gradient of the input tensor using `cudaMemcpy`.
             * - This operation ensures efficient gradient propagation without requiring additional computation.
             *
             * @note
             * - The backward pass assumes that the gradient of the `output` tensor is already computed and properly initialized.
             * - The subtraction operation does not alter the gradient values, enabling a straightforward gradient transfer.
             *
             * @see forward() for the scalar subtraction computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class SubNode
         * @brief Represents a subtraction operation node in a computational graph.
         *
         * The `SubNode` class performs element-wise subtraction between two input tensors. Unlike scalar-based
         * operations, this node handles tensor-to-tensor subtraction, ensuring compatibility of input shapes and
         * propagating gradients for both input tensors during backpropagation.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Computes the element-wise subtraction of two input tensors and stores the result
         *   in the `output` tensor.
         * - **Backward Pass**: Propagates gradients for both input tensors. For the left input tensor, the gradient
         *   is directly copied from the `output` tensor's gradient. For the right input tensor, the gradient is
         *   negated before being propagated.
         * - **Shape Validation**: Ensures the shapes of the two input tensors are identical during construction.
         *   Mismatched shapes result in an exception.
         * - **Gradient Management**: Tracks whether gradients are required for either of the input tensors, and
         *   propagates gradients accordingly.
         *
         * This class is part of the `nz::nodes` namespace and facilitates tensor-to-tensor subtraction
         * operations in computational graphs.
         *
         * @note
         * - The left and right input tensors must have the same shape; otherwise, an exception will be thrown.
         * - Gradients are propagated efficiently, with negation applied to the right input tensor's gradient during backpropagation.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using SubNode for tensor subtraction
         * InputNode input1({3, 3}, true);  // Create the first input node with shape {3, 3}
         * InputNode input2({3, 3}, true);  // Create the second input node with shape {3, 3}
         *
         * input1.output->fill(5.0f);  // Fill the first tensor with value 5.0
         * input2.output->fill(3.0f);  // Fill the second tensor with value 3.0
         *
         * SubNode sub_node(&input1, &input2);  // Subtract input2 from input1
         * sub_node.forward();  // Perform the forward pass
         * sub_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *sub_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the tensor subtraction computation in the forward pass.
         * @see backward() for gradient propagation in the backward pass.
         *
         * @throws std::invalid_argument If the shapes of the two input tensors are not identical.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API SubNode : public Node {
        public:
            /**
             * @brief Constructor to initialize a `SubNode` for tensor subtraction.
             *
             * The constructor initializes a `SubNode`, which performs element-wise subtraction between two input tensors.
             * It validates the shapes of the input tensors to ensure they are compatible for subtraction, establishes
             * connections to the input nodes, and prepares the output tensor for storing the results.
             *
             * @param input_left A pointer to the first input node. Its `output` tensor is treated as the left operand in the subtraction.
             * @param input_right A pointer to the second input node. Its `output` tensor is treated as the right operand in the subtraction.
             *
             * @details
             * - The constructor validates that the shapes of the two input tensors are identical. If the shapes do not match,
             *   an exception is thrown to prevent invalid operations.
             * - The input nodes are added to the `inputs` vector, establishing their connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensors, and its gradient tracking is determined
             *   based on the requirements of the input tensors.
             * - The node's type is set to `"Sub"` to reflect its operation.
             *
             * @note
             * - The input tensors must have the same shape; mismatched shapes result in an exception.
             * - This node supports automatic gradient tracking if either input tensor requires gradients.
             *
             * @throws std::invalid_argument If the shapes of the two input tensors are not identical.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            SubNode(Node* input_left, Node* input_right);

            /**
             * @brief Forward pass for the `SubNode` to perform tensor subtraction.
             *
             * The `forward()` method computes the element-wise subtraction between the two input tensors.
             * It uses a CUDA kernel to perform the operation efficiently on the GPU, storing the result in the `output` tensor.
             *
             * @details
             * - A CUDA kernel (`MatrixSub`) is launched to subtract the elements of the second input tensor
             *   (`inputs[1]`) from the corresponding elements of the first input tensor (`inputs[0]`).
             * - The grid and block dimensions are calculated dynamically based on the size of the `output` tensor
             *   to ensure optimal GPU parallelism.
             * - The result of the subtraction is stored in the `output` tensor.
             *
             * @note
             * - The subtraction operation is performed as `output[i] = input_left[i] - input_right[i]` for each element.
             * - Ensure the shapes of the input tensors are identical; this is enforced during node construction.
             *
             * @see backward() for gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `SubNode` to propagate gradients.
             *
             * The `backward()` method computes and propagates the gradients of the loss with respect to both input tensors.
             * For the left input tensor, the gradient is directly copied from the `output` tensor's gradient. For the right
             * input tensor, the gradient is negated before propagation.
             *
             * @details
             * - If the left input tensor requires gradients, the gradient from the `output` tensor is directly copied to
             *   the gradient tensor of the left input.
             * - If the right input tensor requires gradients, a temporary buffer is allocated, and the gradient of the
             *   `output` tensor is negated using a CUDA kernel (`Negation`) before being copied to the gradient tensor
             *   of the right input.
             * - Temporary GPU memory (`n_grad`) is managed within the method and is released after use.
             *
             * @note
             * - Gradient computation is efficient and ensures proper handling for both input tensors.
             * - The negation step for the right input tensor is necessary due to the derivative of subtraction with respect
             *   to the right operand being -1.
             *
             * @warning
             * - Proper memory management is ensured by freeing the temporary buffer (`n_grad`) after use.
             *
             * @see forward() for the tensor subtraction computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class ReLUNode
         * @brief Represents a Rectified Linear Unit (ReLU) operation node in a computational graph.
         *
         * The `ReLUNode` class applies the ReLU activation function to the input tensor. ReLU is a commonly used
         * non-linear activation function in neural networks, defined as `ReLU(x) = max(0, x)`. It introduces
         * non-linearity and sparsity into the network.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Applies the ReLU activation function element-wise to the input tensor.
         *   Values less than zero are set to zero, while non-negative values remain unchanged.
         * - **Backward Pass**: Computes the gradient of the loss with respect to the input tensor. Gradients
         *   are passed through unchanged for positive input values and set to zero for negative input values.
         * - **Shape Preservation**: The output tensor has the same shape as the input tensor.
         * - **Gradient Management**: Automatically tracks gradients if required by the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is typically used in constructing neural
         * network models to introduce non-linearity between layers.
         *
         * @note
         * - The input tensor's shape is preserved in the output tensor.
         * - Gradients are efficiently computed using CUDA kernels.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using ReLUNode in a computational graph
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         *
         * float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};  // Sample input values
         * input.output->dataInject(data);  // Copy data to the input tensor
         *
         * ReLUNode relu_node(&input);  // Apply ReLU activation
         * relu_node.forward();  // Perform the forward pass
         * relu_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *relu_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the ReLU activation computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API ReLUNode : public Node {
        public:
            /**
             * @brief Constructor to initialize a `ReLUNode` for applying the ReLU activation function.
             *
             * The constructor initializes a `ReLUNode`, which applies the Rectified Linear Unit (ReLU) activation
             * function to an input tensor. It establishes a connection to the input node, initializes the output
             * tensor, and sets the type of the node to "ReLU".
             *
             * @param input A pointer to the input node. Its `output` tensor will have the ReLU activation applied.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and its gradient tracking
             *   is determined based on the input tensor's requirements.
             * - The node's type is set to "ReLU" to reflect its operation.
             *
             * @note
             * - The ReLU activation is defined as `ReLU(x) = max(0, x)`. This will be applied during the forward pass.
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient propagation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            explicit ReLUNode(Node* input);

            /**
             * @brief Forward pass for the `ReLUNode` to apply the ReLU activation function.
             *
             * The `forward()` method applies the Rectified Linear Unit (ReLU) activation function element-wise to the
             * input tensor. Values less than zero are set to zero, while non-negative values remain unchanged. The
             * results are stored in the `output` tensor.
             *
             * @details
             * - A CUDA kernel (`RectifiedLinearUnit`) is launched to compute the ReLU activation in parallel on the GPU.
             * - The grid and block dimensions are calculated dynamically based on the size of the `output` tensor
             *   to optimize GPU performance.
             * - The output tensor stores the result of applying `ReLU(x) = max(0, x)` for each element of the input tensor.
             *
             * @note
             * - The shape of the output tensor matches that of the input tensor.
             * - Ensure the input tensor is properly initialized before calling this method.
             *
             * @see backward() for gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `ReLUNode` to compute gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor by applying
             * the derivative of the ReLU activation function. Gradients are propagated only for elements where the
             * input tensor values are positive; otherwise, the gradients are set to zero.
             *
             * @details
             * - A CUDA kernel (`ReLUBackward`) is launched to compute the gradients in parallel on the GPU.
             * - The grid and block dimensions are calculated dynamically based on the size of the `output` tensor.
             * - The derivative of ReLU is defined as:
             *   ```
             *   ReLU'(x) = 1, if x > 0
             *             0, if x <= 0
             *   ```
             *   This derivative is applied element-wise to propagate gradients through the ReLU operation.
             * - Gradients are accumulated in the gradient tensor of the input node.
             *
             * @note
             * - Gradients are only propagated if the input tensor's `requiresGrad` property is true.
             * - The shape of the gradient tensor matches the shape of the input tensor.
             *
             * @see forward() for the ReLU activation computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class SigmoidNode
         * @brief Represents a Sigmoid activation function node in a computational graph.
         *
         * The `SigmoidNode` class applies the Sigmoid activation function to the input tensor. The Sigmoid
         * function is defined as:
         * ```
         * Sigmoid(x) = 1 / (1 + exp(-x))
         * ```
         * It is commonly used in neural networks for binary classification tasks or as a gating mechanism
         * in recurrent networks.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Applies the Sigmoid activation function element-wise to the input tensor, mapping
         *   values to the range (0, 1).
         * - **Backward Pass**: Computes the gradient of the loss with respect to the input tensor using the derivative
         *   of the Sigmoid function, which is:
         *   ```
         *   Sigmoid'(x) = Sigmoid(x) * (1 - Sigmoid(x))
         *   ```
         * - **Shape Preservation**: The output tensor has the same shape as the input tensor.
         * - **Gradient Management**: Automatically tracks gradients if required by the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is used to add non-linearity to models or
         * normalize outputs.
         *
         * @note
         * - The Sigmoid function is applied element-wise, and the output values are restricted to the range (0, 1).
         * - Efficient GPU computations are performed for both forward and backward passes.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using SigmoidNode in a computational graph
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         *
         * float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};  // Sample input values
         * input.output->dataInject(data);  // Copy data to the input tensor
         *
         * SigmoidNode sigmoid_node(&input);  // Apply Sigmoid activation
         * sigmoid_node.forward();  // Perform the forward pass
         * sigmoid_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *sigmoid_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the Sigmoid activation computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API SigmoidNode : public Node {
        public:
            /**
             * @brief Constructor to initialize a `SigmoidNode` for applying the Sigmoid activation function.
             *
             * The constructor initializes a `SigmoidNode`, which applies the Sigmoid activation function to an input tensor.
             * It establishes a connection to the input node, initializes the output tensor, and sets the type of the node
             * to "Sigmoid".
             *
             * @param input A pointer to the input node. Its `output` tensor will have the Sigmoid activation applied.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and its gradient tracking
             *   is determined based on the input tensor's requirements.
             * - The node's type is set to "Sigmoid" to reflect its operation.
             *
             * @note
             * - The Sigmoid activation function maps input values to the range (0, 1) and is defined as:
             *   ```
             *   Sigmoid(x) = 1 / (1 + exp(-x))
             *   ```
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            explicit SigmoidNode(Node* input);

            /**
             * @brief Forward pass for the `SigmoidNode` to apply the Sigmoid activation function.
             *
             * The `forward()` method applies the Sigmoid activation function element-wise to the input tensor.
             * The result is stored in the `output` tensor. The Sigmoid function is defined as:
             * ```
             * Sigmoid(x) = 1 / (1 + exp(-x))
             * ```
             * It maps input values to the range (0, 1).
             *
             * @details
             * - A CUDA kernel (`Sigmoid`) is launched to compute the activation function in parallel on the GPU.
             * - The grid and block dimensions are dynamically calculated based on the size of the `output` tensor
             *   to ensure efficient GPU utilization.
             * - The computed values are stored in the `output` tensor for use in subsequent layers or operations.
             *
             * @see backward() for the computation of gradients in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `SigmoidNode` to compute gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor
             * by applying the derivative of the Sigmoid activation function. The gradient is propagated
             * using the formula:
             * ```
             * Sigmoid'(x) = Sigmoid(x) * (1 - Sigmoid(x))
             * ```
             *
             * @details
             * - A CUDA kernel (`SigmoidBackward`) is launched to compute the gradients in parallel on the GPU.
             * - The derivative of the Sigmoid function is applied element-wise to the `output` tensor's data
             *   and combined with the gradient of the `output` tensor to compute the input gradient.
             * - The computed gradient is stored in the gradient tensor of the input node.
             *
             * @note
             * - Gradients are only computed and propagated if the input tensor's `requiresGrad` property is true.
             * - The shape of the gradient tensor matches that of the input tensor.
             *
             * @see forward() for the Sigmoid activation computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class TanhNode
         * @brief Represents a hyperbolic tangent (tanh) activation function node in a computational graph.
         *
         * The `TanhNode` class applies the hyperbolic tangent (tanh) activation function to the input tensor.
         * The tanh function is defined as:
         * ```
         * Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
         * ```
         * It maps input values to the range (-1, 1) and is commonly used in neural networks for non-linear activation.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Applies the tanh activation function element-wise to the input tensor, mapping values
         *   to the range (-1, 1).
         * - **Backward Pass**: Computes the gradient of the loss with respect to the input tensor using the derivative
         *   of the tanh function, which is:
         *   ```
         *   Tanh'(x) = 1 - Tanh(x)^2
         *   ```
         * - **Shape Preservation**: The output tensor has the same shape as the input tensor.
         * - **Gradient Management**: Automatically tracks gradients if required by the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is commonly used to add non-linearity
         * to models or normalize outputs.
         *
         * @note
         * - The tanh function is applied element-wise, mapping the input tensor to the range (-1, 1).
         * - Efficient GPU computations are performed for both forward and backward passes.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using TanhNode in a computational graph
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         *
         * float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};  // Sample input values
         * input.output->dataInject(data);  // Copy data to the input tensor
         *
         * TanhNode tanh_node(&input);  // Apply tanh activation
         * tanh_node.forward();  // Perform the forward pass
         * tanh_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *tanh_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the tanh activation computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API TanhNode : public Node {
        public:
            /**
             * @brief Constructor to initialize a `TanhNode` for applying the tanh activation function.
             *
             * The constructor initializes a `TanhNode`, which applies the hyperbolic tangent (tanh) activation
             * function to an input tensor. It establishes a connection to the input node, initializes the output
             * tensor, and sets the type of the node to "Tanh".
             *
             * @param input A pointer to the input node. Its `output` tensor will have the tanh activation applied.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and its gradient tracking
             *   is determined based on the input tensor's requirements.
             * - The node's type is set to "Tanh" to reflect its operation.
             *
             * @note
             * - The tanh activation function maps input values to the range (-1, 1) and is defined as:
             *   ```
             *   Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
             *   ```
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            explicit TanhNode(Node* input);

            /**
             * @brief Forward pass for the `TanhNode` to apply the tanh activation function.
             *
             * The `forward()` method applies the hyperbolic tangent (tanh) activation function element-wise to
             * the input tensor. The result is stored in the `output` tensor. The tanh function is defined as:
             * ```
             * Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
             * ```
             * It maps input values to the range (-1, 1).
             *
             * @details
             * - A CUDA kernel (`Tanh`) is launched to compute the activation function in parallel on the GPU.
             * - The grid and block dimensions are dynamically calculated based on the size of the `output` tensor
             *   to optimize GPU utilization.
             * - The computed values are stored in the `output` tensor for use in subsequent layers or operations.
             *
             * @see backward() for the computation of gradients in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `TanhNode` to compute gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor
             * by applying the derivative of the tanh activation function. The gradient is propagated using the formula:
             * ```
             * Tanh'(x) = 1 - Tanh(x)^2
             * ```
             *
             * @details
             * - A CUDA kernel (`TanhBackward`) is launched to compute the gradients in parallel on the GPU.
             * - The derivative of the tanh function is applied element-wise to the `output` tensor's data
             *   and combined with the gradient of the `output` tensor to compute the input gradient.
             * - The computed gradient is stored in the gradient tensor of the input node.
             *
             * @note
             * - Gradients are only computed and propagated if the input tensor's `requiresGrad` property is true.
             * - The shape of the gradient tensor matches that of the input tensor.
             *
             * @see forward() for the tanh activation computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class LeakyReLUNode
         * @brief Represents a Leaky Rectified Linear Unit (LeakyReLU) activation function node in a computational graph.
         *
         * The `LeakyReLUNode` class applies the Leaky ReLU activation function to the input tensor. Unlike the standard
         * ReLU, Leaky ReLU allows a small, non-zero gradient for negative input values. The function is defined as:
         * ```
         * LeakyReLU(x) = x, if x > 0
         *                alpha * x, if x <= 0
         * ```
         * where `alpha` is a small constant that determines the slope for negative values.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Applies the Leaky ReLU activation function element-wise to the input tensor. Positive
         *   values remain unchanged, while negative values are scaled by `alpha`.
         * - **Backward Pass**: Computes the gradient of the loss with respect to the input tensor. Gradients for
         *   positive values are passed through unchanged, while gradients for negative values are scaled by `alpha`.
         * - **Shape Preservation**: The output tensor has the same shape as the input tensor.
         * - **Gradient Management**: Automatically tracks gradients if required by the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is commonly used in deep learning to mitigate
         * the "dying ReLU" problem by allowing small gradients for negative inputs.
         *
         * @note
         * - The `alpha` parameter defaults to `0.01`, but can be customized during construction.
         * - Efficient GPU computations are performed for both forward and backward passes.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using LeakyReLUNode in a computational graph
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         *
         * float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};  // Sample input values
         * input.output->dataInject(data);  // Copy data to the input tensor
         *
         * LeakyReLUNode leaky_relu_node(&input, 0.1f);  // Apply Leaky ReLU activation with alpha = 0.1
         * leaky_relu_node.forward();  // Perform the forward pass
         * leaky_relu_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *leaky_relu_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the Leaky ReLU computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API LeakyReLUNode : public Node {
            Tensor::value_type alpha;

        public:
            /**
             * @brief Constructor to initialize a `LeakyReLUNode` for applying the Leaky ReLU activation function.
             *
             * The constructor initializes a `LeakyReLUNode`, which applies the Leaky ReLU activation function to
             * an input tensor. It establishes a connection to the input node, initializes the output tensor, and
             * sets the `alpha` parameter and node type.
             *
             * @param input A pointer to the input node. Its `output` tensor will have the Leaky ReLU activation applied.
             * @param alpha The slope for negative input values, determining how much they are scaled. Defaults to `0.01`.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and its gradient tracking
             *   is determined based on the input tensor's requirements.
             * - The `alpha` parameter is stored to control the slope for negative input values during both the forward
             *   and backward passes.
             * - The node's type is set to "LeakyReLU" to reflect its operation.
             *
             * @note
             * - The Leaky ReLU activation function is defined as:
             *   ```
             *   LeakyReLU(x) = x, if x > 0
             *                  alpha * x, if x <= 0
             *   ```
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            explicit LeakyReLUNode(Node* input, Tensor::value_type alpha = 0.01f);

            /**
             * @brief Forward pass for the `LeakyReLUNode` to apply the Leaky ReLU activation function.
             *
             * The `forward()` method applies the Leaky ReLU activation function element-wise to the input tensor.
             * Positive input values remain unchanged, while negative input values are scaled by the `alpha` parameter.
             * The result is stored in the `output` tensor. The function is defined as:
             * ```
             * LeakyReLU(x) = x, if x > 0
             *                alpha * x, if x <= 0
             * ```
             *
             * @details
             * - A CUDA kernel (`LeakyReLU`) is launched to compute the activation function in parallel on the GPU.
             * - The grid and block dimensions are dynamically calculated based on the size of the `output` tensor
             *   to ensure efficient GPU utilization.
             * - The `alpha` parameter, provided during construction, determines the slope for negative input values.
             *
             * @see backward() for the computation of gradients in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `LeakyReLUNode` to compute gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor
             * by applying the derivative of the Leaky ReLU activation function. The gradient computation is
             * defined as:
             * ```
             * LeakyReLU'(x) = 1, if x > 0
             *                 alpha, if x <= 0
             * ```
             *
             * @details
             * - A CUDA kernel (`LeakyReLUBackward`) is launched to compute the gradients in parallel on the GPU.
             * - The derivative of the Leaky ReLU function is applied element-wise to the input tensor's data
             *   and combined with the gradient of the `output` tensor to compute the input gradient.
             * - The `alpha` parameter, provided during construction, controls the gradient scale for negative input values.
             * - The computed gradient is stored in the gradient tensor of the input node.
             *
             * @note
             * - Gradients are only computed and propagated if the input tensor's `requiresGrad` property is true.
             * - The shape of the gradient tensor matches that of the input tensor.
             *
             * @see forward() for the Leaky ReLU computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class SwishNode
         * @brief Represents a Swish activation function node in a computational graph.
         *
         * The `SwishNode` class applies the Swish activation function to the input tensor. The Swish function
         * is defined as:
         * ```
         * Swish(x) = x / (1 + exp(-x))
         * ```
         * It is a smooth, non-monotonic activation function that often outperforms ReLU in deep learning tasks.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Applies the Swish activation function element-wise to the input tensor,
         *   blending the input with its sigmoid output.
         * - **Backward Pass**: Computes the gradient of the loss with respect to the input tensor using the
         *   derivative of the Swish function:
         *   ```
         *   Swish'(x) = Sigmoid(x) + x * Sigmoid(x) * (1 - Sigmoid(x))
         *   ```
         * - **Shape Preservation**: The output tensor has the same shape as the input tensor.
         * - **Gradient Management**: Automatically tracks gradients if required by the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is commonly used in advanced deep learning
         * models to enhance performance over traditional activation functions.
         *
         * @note
         * - The Swish function is applied element-wise, and it smoothly maps input values while retaining non-linearity.
         * - Efficient GPU computations are performed for both forward and backward passes.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using SwishNode in a computational graph
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         *
         * float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};  // Sample input values
         * input.output->dataInject(data);  // Copy data to the input tensor
         *
         * SwishNode swish_node(&input);  // Apply Swish activation
         * swish_node.forward();  // Perform the forward pass
         * swish_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *swish_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the Swish computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API SwishNode : public Node {
        public:
            /**
             * @brief Constructor to initialize a `SwishNode` for applying the Swish activation function.
             *
             * The constructor initializes a `SwishNode`, which applies the Swish activation function to an input tensor.
             * It establishes a connection to the input node, initializes the output tensor, and sets the type of the node
             * to "Swish".
             *
             * @param input A pointer to the input node. Its `output` tensor will have the Swish activation applied.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and its gradient tracking
             *   is determined based on the input tensor's requirements.
             * - The node's type is set to "Swish" to reflect its operation.
             *
             * @note
             * - The Swish activation function is defined as:
             *   ```
             *   Swish(x) = x / (1 + exp(-x))
             *   ```
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            explicit SwishNode(Node* input);

            /**
             * @brief Forward pass for the `SwishNode` to apply the Swish activation function.
             *
             * The `forward()` method applies the Swish activation function element-wise to the input tensor.
             * The result is stored in the `output` tensor. The Swish function is defined as:
             * ```
             * Swish(x) = x / (1 + exp(-x))
             * ```
             *
             * @details
             * - A CUDA kernel (`Swish`) is launched to compute the activation function in parallel on the GPU.
             * - The grid and block dimensions are dynamically calculated based on the size of the `output` tensor
             *   to optimize GPU performance.
             * - The computed values are stored in the `output` tensor for use in subsequent layers or operations.
             *
             * @see backward() for the computation of gradients in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `SwishNode` to compute gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor by applying
             * the derivative of the Swish activation function. The gradient computation is based on the formula:
             * ```
             * Swish'(x) = Sigmoid(x) + x * Sigmoid(x) * (1 - Sigmoid(x))
             * ```
             * where `Sigmoid(x) = 1 / (1 + exp(-x))`.
             *
             * @details
             * - A CUDA kernel (`SwishBackward`) is launched to compute the gradients in parallel on the GPU.
             * - The derivative of the Swish function is applied element-wise to the input tensor's data
             *   and combined with the gradient of the `output` tensor to compute the input gradient.
             * - The computed gradient is stored in the gradient tensor of the input node.
             *
             * @note
             * - Gradients are only computed and propagated if the input tensor's `requiresGrad` property is true.
             * - The shape of the gradient tensor matches that of the input tensor.
             *
             * @see forward() for the Swish computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class ELUNode
         * @brief Represents an Exponential Linear Unit (ELU) activation function node in a computational graph.
         *
         * The `ELUNode` class applies the Exponential Linear Unit (ELU) activation function to the input tensor.
         * The ELU function is defined as:
         * ```
         * ELU(x) = x, if x > 0
         *          alpha * (exp(x) - 1), if x <= 0
         * ```
         * where `alpha` controls the value for negative inputs and smoothens the curve.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Applies the ELU activation function element-wise to the input tensor. Positive
         *   values remain unchanged, while negative values are scaled exponentially with `alpha`.
         * - **Backward Pass**: Computes the gradient of the loss with respect to the input tensor. Gradients
         *   are propagated differently for positive and negative input values:
         *   ```
         *   ELU'(x) = 1, if x > 0
         *             alpha * exp(x), if x <= 0
         *   ```
         * - **Shape Preservation**: The output tensor has the same shape as the input tensor.
         * - **Gradient Management**: Automatically tracks gradients if required by the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is often used to improve the learning dynamics
         * in deep networks by reducing the vanishing gradient problem for negative inputs.
         *
         * @note
         * - The `alpha` parameter defaults to `1.0`, but can be customized during construction to control the behavior
         *   for negative inputs.
         * - Efficient GPU computations are performed for both forward and backward passes.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using ELUNode in a computational graph
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         *
         * float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};  // Sample input values
         * input.output->dataInject(data);  // Copy data to the input tensor
         *
         * ELUNode elu_node(&input, 0.5f);  // Apply ELU activation with alpha = 0.5
         * elu_node.forward();  // Perform the forward pass
         * elu_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *elu_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the ELU computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API ELUNode : public Node {
            Tensor::value_type alpha;

        public:
            /**
             * @brief Constructor to initialize an `ELUNode` for applying the ELU activation function.
             *
             * The constructor initializes an `ELUNode`, which applies the Exponential Linear Unit (ELU) activation
             * function to an input tensor. It establishes a connection to the input node, initializes the output tensor,
             * and sets the `alpha` parameter and node type.
             *
             * @param input A pointer to the input node. Its `output` tensor will have the ELU activation applied.
             * @param alpha The scaling parameter for negative input values. Defaults to `1.0`.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and its gradient tracking
             *   is determined based on the input tensor's requirements.
             * - The `alpha` parameter controls the scaling for negative input values, influencing the gradient flow
             *   and smoothness of the activation.
             * - The node's type is set to "ELU" to reflect its operation.
             *
             * @note
             * - The ELU activation function is defined as:
             *   ```
             *   ELU(x) = x, if x > 0
             *            alpha * (exp(x) - 1), if x <= 0
             *   ```
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            explicit ELUNode(Node* input, Tensor::value_type alpha = 1.0f);

            /**
             * @brief Forward pass for the `ELUNode` to apply the ELU activation function.
             *
             * The `forward()` method applies the Exponential Linear Unit (ELU) activation function element-wise
             * to the input tensor. The result is stored in the `output` tensor. The ELU function is defined as:
             * ```
             * ELU(x) = x, if x > 0
             *          alpha * (exp(x) - 1), if x <= 0
             * ```
             *
             * @details
             * - A CUDA kernel (`ExponentialLinearUnit`) is launched to compute the activation function in parallel on the GPU.
             * - The grid and block dimensions are dynamically calculated based on the size of the `output` tensor
             *   to ensure efficient GPU utilization.
             * - The `alpha` parameter, provided during construction, scales the output for negative input values.
             *
             * @see backward() for the computation of gradients in the backward pass.
             *
             * @note
             * - The shape of the output tensor matches that of the input tensor.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `ELUNode` to compute gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor
             * by applying the derivative of the ELU activation function. The gradient computation is based on the formula:
             * ```
             * ELU'(x) = 1, if x > 0
             *           alpha * exp(x), if x <= 0
             * ```
             * where `alpha` is the scaling parameter for negative input values.
             *
             * @details
             * - A CUDA kernel (`ELUBackward`) is launched to compute the gradients in parallel on the GPU.
             * - The derivative of the ELU function is applied element-wise to the input tensor's data
             *   and combined with the gradient of the `output` tensor to compute the input gradient.
             * - The `alpha` parameter, provided during construction, controls the gradient scale for negative input values.
             * - The computed gradient is stored in the gradient tensor of the input node.
             *
             * @note
             * - Gradients are only computed and propagated if the input tensor's `requiresGrad` property is true.
             * - The shape of the gradient tensor matches that of the input tensor.
             *
             * @see forward() for the ELU computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class HardSigmoidNode
         * @brief Represents a Hard Sigmoid activation function node in a computational graph.
         *
         * The `HardSigmoidNode` class applies the Hard Sigmoid activation function to the input tensor.
         * The Hard Sigmoid function is a computationally efficient approximation of the sigmoid function
         * and is defined as:
         * ```
         * HardSigmoid(x) = max(0, min(1, alpha * x + beta))
         * ```
         * where `alpha` and `beta` control the slope and offset, respectively.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Applies the Hard Sigmoid activation function element-wise to the input tensor,
         *   mapping values to the range [0, 1] with linear interpolation.
         * - **Backward Pass**: Computes the gradient of the loss with respect to the input tensor. Gradients
         *   are propagated only for input values within the linear range (`0 <= alpha * x + beta <= 1`):
         *   ```
         *   HardSigmoid'(x) = alpha, if 0 <= alpha * x + beta <= 1
         *                     0, otherwise
         *   ```
         * - **Shape Preservation**: The output tensor has the same shape as the input tensor.
         * - **Gradient Management**: Automatically tracks gradients if required by the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is used in models where efficiency is prioritized
         * over precise non-linearity.
         *
         * @note
         * - The `alpha` and `beta` parameters default to `0.2` and `0.5`, respectively, but can be customized during construction.
         * - Efficient GPU computations are performed for both forward and backward passes.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using HardSigmoidNode in a computational graph
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         *
         * float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};  // Sample input values
         * input.output->dataInject(data);  // Copy data to the input tensor
         *
         * HardSigmoidNode hard_sigmoid_node(&input, 0.2f, 0.5f);  // Apply Hard Sigmoid activation
         * hard_sigmoid_node.forward();  // Perform the forward pass
         * hard_sigmoid_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *hard_sigmoid_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the Hard Sigmoid computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API HardSigmoidNode : public Node {
            Tensor::value_type alpha;
            Tensor::value_type beta;

        public:
            /**
             * @brief Constructor to initialize a `HardSigmoidNode` for applying the Hard Sigmoid activation function.
             *
             * The constructor initializes a `HardSigmoidNode`, which applies the Hard Sigmoid activation function to
             * an input tensor. It establishes a connection to the input node, initializes the output tensor, and sets
             * the `alpha` and `beta` parameters as well as the node type.
             *
             * @param input A pointer to the input node. Its `output` tensor will have the Hard Sigmoid activation applied.
             * @param alpha The slope parameter for the linear part of the Hard Sigmoid function. Defaults to `0.2`.
             * @param beta The offset parameter for the Hard Sigmoid function. Defaults to `0.5`.
             *
             * @details
             * - The input node is added to the `inputs` vector to establish the connection in the computational graph.
             * - The `output` tensor is initialized with the same shape as the input tensor, and its gradient tracking
             *   is determined based on the input tensor's requirements.
             * - The `alpha` and `beta` parameters control the slope and offset of the Hard Sigmoid activation function,
             *   influencing the gradient flow and the range mapping.
             * - The node's type is set to "HardSigmoid" to reflect its operation.
             *
             * @note
             * - The Hard Sigmoid activation function is defined as:
             *   ```
             *   HardSigmoid(x) = max(0, min(1, alpha * x + beta))
             *   ```
             * - This node supports automatic gradient tracking if the input tensor requires gradients.
             *
             * @see forward() for the forward pass implementation.
             * @see backward() for gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            explicit HardSigmoidNode(Node* input, Tensor::value_type alpha = 0.2f, Tensor::value_type beta = 0.5f);

            /**
             * @brief Forward pass for the `HardSigmoidNode` to apply the Hard Sigmoid activation function.
             *
             * The `forward()` method applies the Hard Sigmoid activation function element-wise to the input tensor.
             * The result is stored in the `output` tensor. The Hard Sigmoid function is defined as:
             * ```
             * HardSigmoid(x) = max(0, min(1, alpha * x + beta))
             * ```
             *
             * @details
             * - A CUDA kernel (`HardSigmoid`) is launched to compute the activation function in parallel on the GPU.
             * - The grid and block dimensions are dynamically calculated based on the size of the `output` tensor
             *   to ensure efficient GPU utilization.
             * - The `alpha` and `beta` parameters, provided during construction, control the slope and offset of
             *   the linear part of the activation function.
             *
             * @note
             * - The shape of the output tensor matches that of the input tensor.
             *
             * @see backward() for the computation of gradients in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void forward() override;

            /**
             * @brief Backward pass for the `HardSigmoidNode` to compute gradients.
             *
             * The `backward()` method computes the gradient of the loss with respect to the input tensor
             * by applying the derivative of the Hard Sigmoid activation function. The gradient computation is defined as:
             * ```
             * HardSigmoid'(x) = alpha, if 0 <= alpha * x + beta <= 1
             *                   0, otherwise
             * ```
             * where `alpha` and `beta` control the slope and offset of the Hard Sigmoid function.
             *
             * @details
             * - A CUDA kernel (`HardSigmoidBackward`) is launched to compute the gradients in parallel on the GPU.
             * - The derivative of the Hard Sigmoid function is applied element-wise to the input tensor's data
             *   and combined with the gradient of the `output` tensor to compute the input gradient.
             * - The computed gradient is stored in the gradient tensor of the input node.
             *
             * @note
             * - Gradients are only computed and propagated if the input tensor's `requiresGrad` property is true.
             * - The shape of the gradient tensor matches that of the input tensor.
             *
             * @see forward() for the Hard Sigmoid computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            void backward() override;
        };

        /**
         * @class HardSwishNode
         * @brief Represents a Hard Swish activation function node in a computational graph.
         *
         * The `HardSwishNode` class applies the Hard Swish activation function to the input tensor.
         * The Hard Swish function is a computationally efficient approximation of the Swish function
         * and is defined as:
         * ```
         * HardSwish(x) = x * max(0, min(1, alpha * x + beta))
         * ```
         * where `alpha` and `beta` control the slope and offset of the linear part of the function.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Applies the Hard Swish activation function element-wise to the input tensor,
         *   blending the input with a clipped linear function.
         * - **Backward Pass**: Computes the gradient of the loss with respect to the input tensor, handling
         *   linear and non-linear regions separately.
         * - **Shape Preservation**: The output tensor has the same shape as the input tensor.
         * - **Gradient Management**: Automatically tracks gradients if required by the input tensor.
         *
         * This class is part of the `nz::nodes` namespace and is used in models to improve performance
         * while maintaining computational efficiency.
         *
         * @note
         * - The `alpha` and `beta` parameters default to `1.0` and `0.5`, respectively, but can be customized during construction.
         * - Efficient GPU computations are performed for both forward and backward passes.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using HardSwishNode in a computational graph
         * InputNode input({3, 3}, true);  // Create an input node with shape {3, 3}
         *
         * float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};  // Sample input values
         * input.output->dataInject(data);  // Copy data to the input tensor
         *
         * HardSwishNode hard_swish_node(&input, 1.0f, 0.5f);  // Apply Hard Swish activation
         * hard_swish_node.forward();  // Perform the forward pass
         * hard_swish_node.backward();  // Propagate gradients in the backward pass
         *
         * std::cout << "Output: " << *hard_swish_node.output << std::endl;  // Print the result
         * ```
         *
         * @see forward() for the Hard Swish computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/05
         */
        class DL_API HardSwishNode : public Node {
            Tensor::value_type alpha;
            Tensor::value_type beta;

        public:
            /**
             * @brief Constructor to initialize a `HardSwishNode` for applying the Hard Swish activation function.
             *
             * The constructor initializes a `HardSwishNode`, which applies the Hard Swish activation function to
             * an input tensor. It establishes a connection to the input node, initializes the output tensor, and sets
             * the `alpha` and `beta` parameters as well as the node type.
             *
             * @param input A pointer to the input node. Its `output` tensor will have the Hard Swish activation applied.
             * @param alpha The slope parameter for the Hard Swish function. Controls the steepness of the curve.
             * @param beta The offset parameter for the Hard Swish function. Shifts the function horizontally.
             *
             * @details
             * The Hard Swish activation function is defined as:
             * ```
             * HardSwish(x) = x * max(0, min(1, alpha * x + beta))
             * ```
             *
             * Key operations performed by the constructor:
             * - Adds the input node to the `inputs` vector, establishing the connection in the computational graph.
             * - Determines if gradient tracking is required based on the input tensor's `requiresGrad` property.
             * - Initializes the `output` tensor with the same shape as the input tensor and appropriate gradient tracking.
             * - Sets the `alpha` and `beta` parameters, which control the shape of the Hard Swish function.
             * - Sets the node type to "HardSwish" for identification in the computational graph.
             *
             * @note
             * - The Hard Swish function is a smooth approximation of the ReLU activation, combining properties of
             *   ReLU and Swish activations.
             * - The `alpha` and `beta` parameters allow for customization of the activation function's behavior.
             * - Gradient tracking for the output tensor is automatically set based on the input tensor's requirements.
             *
             * @see forward() for the implementation of the forward pass using these parameters.
             * @see backward() for the gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/05
             */
            explicit HardSwishNode(Node* input, Tensor::value_type alpha = 1.0f, Tensor::value_type beta = 0.5f);

            /**
             * @brief Forward pass for the `HardSwishNode` to apply the Hard Swish activation function.
             *
             * This method implements the forward pass of the Hard Swish activation function. It applies
             * the Hard Swish operation element-wise to the input tensor and stores the result in the output tensor.
             *
             * @details
             * The Hard Swish function is defined as:
             * ```
             * HardSwish(x) = x * max(0, min(1, alpha * x + beta))
             * ```
             * where `alpha` and `beta` are parameters that control the shape of the function.
             *
             * Key operations:
             * - Configures CUDA execution parameters (grid and block dimensions) for parallel processing.
             * - Launches a CUDA kernel (`HardSwish`) to perform the Hard Swish computation on the GPU.
             * - Processes all elements of the input tensor in parallel.
             *
             * CUDA kernel configuration:
             * - Block size: 256 threads per block.
             * - Grid size: Calculated to ensure coverage of all elements in the output tensor.
             *
             * @note
             * - This method assumes that the CUDA kernel `HardSwish` is defined elsewhere and properly
             *   implements the Hard Swish function.
             * - The output tensor is assumed to have the same shape as the input tensor.
             * - This implementation leverages GPU parallelism for efficient computation, especially
             *   for large tensors.
             *
             * @see backward() for the corresponding backward pass implementation.
             * @see HardSwishNode constructor for the initialization of `alpha` and `beta` parameters.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/5
             */
            void forward() override;

            /**
             * @brief Backward pass for the `HardSwishNode` to compute gradients.
             *
             * This method implements the backward pass of the Hard Swish activation function. It computes
             * the gradient of the loss with respect to the input by applying the derivative of the Hard Swish
             * function to the incoming gradient.
             *
             * @details
             * The derivative of the Hard Swish function is:
             * ```
             * HardSwish'(x) = (2 * alpha * x + beta) * min(max(alpha * x + beta, 0), 1) +
             *                 max(0, min(1, alpha * x + beta)) +
             *                 x * (alpha * (x > -beta/alpha) * (x < (1-beta)/alpha))
             * ```
             * where `alpha` and `beta` are the parameters that control the shape of the function.
             *
             * Key operations:
             * - Checks if the input tensor requires gradient computation.
             * - If gradients are required:
             *   - Configures CUDA execution parameters (grid and block dimensions) for parallel processing.
             *   - Launches a CUDA kernel (`HardSwishBackward`) to compute gradients on the GPU.
             *   - Processes all elements of the input tensor in parallel.
             *
             * CUDA kernel configuration:
             * - Block size: 256 threads per block.
             * - Grid size: Calculated to ensure coverage of all elements in the input tensor.
             *
             * @note
             * - This method is only executed if the input tensor requires gradient computation.
             * - The method assumes that the CUDA kernel `HardSwishBackward` is defined elsewhere and correctly
             *   implements the derivative of the Hard Swish function.
             * - The gradient computation leverages GPU parallelism for efficiency, especially for large tensors.
             * - The computed gradients are accumulated in the input tensor's gradient buffer.
             *
             * @see forward() for the corresponding forward pass implementation.
             * @see HardSwishNode constructor for the initialization of `alpha` and `beta` parameters.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/5
             */
            void backward() override;
        };

        /**
         * @class SoftmaxNode
         * @brief Implements the Softmax activation function as a node in a neural network computational graph.
         *
         * The `SoftmaxNode` class applies the Softmax activation function to the input tensor, transforming
         * it into a probability distribution. This node is commonly used as the final layer in classification
         * networks to convert raw scores into probabilities.
         *
         * The Softmax function is defined as:
         * ```
         * Softmax(x_i) = exp(x_i) / sum(exp(x_j))
         * ```
         * where x_i is the i-th element of the input vector and the sum is over all elements j.
         *
         * @details
         * Key features and characteristics:
         * - **Probability Output**: Transforms input into a probability distribution where all elements sum to 1.
         * - **Numerically Stable**: Implements a numerically stable version of Softmax to prevent overflow.
         * - **Shape Preservation**: The output tensor maintains the same shape as the input tensor.
         * - **GPU Acceleration**: Utilizes CUDA for efficient parallel computation on GPU.
         * - **Gradient Computation**: Supports backward pass for gradient calculation in neural network training.
         * - **Precomputation Optimization**: Precomputes exponential sum in the constructor for efficiency.
         *
         * Implementation details:
         * - The constructor precomputes the sum of exponentials to optimize the forward pass.
         * - The forward pass applies the Softmax function using the precomputed sum.
         * - The backward pass computes the full Jacobian matrix for accurate gradient calculation.
         * - CUDA kernels are used for parallel computation in both forward and backward passes.
         *
         * Use cases:
         * - Output layer of multi-class classification networks.
         * - Attention mechanisms in sequence-to-sequence models.
         * - Any scenario requiring normalization of a vector into a probability distribution.
         *
         * Limitations and considerations:
         * - May suffer from underflow or overflow for extreme input values.
         * - The full Jacobian computation in backward pass can be memory-intensive for large outputs.
         *
         * @note
         * - This implementation assumes the input is a 1D or 2D tensor. For higher dimensions, consider using a
         *   dimension-specific Softmax implementation.
         * - The node automatically handles gradient tracking based on the input tensor's requirements.
         * - For very large inputs, consider using LogSoftmax for improved numerical stability.
         *
         * @see forward() for the Softmax computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * ### Usage Example:
         * ```cpp
         * // Creating a Softmax node in a neural network
         * InputNode input({1, 1, 1, 5}, true);  // Input node with shape {1, 1, 1, 5}
         * std::vector<float> logits{2.0f, 1.0f, 0.1f, 3.0f, -1.0f};
         * input.output->dataInject(logits.begin(), logits.end());
         *
         * SoftmaxNode softmax(&input);
         * softmax.forward();
         *
         * // The output tensor now contains the probability distribution
         * std::cout << "Probabilities: " << *softmax.output << std::endl;
         *
         * // Backward pass for gradient computation
         * softmax.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/5
         */
        class DL_API SoftmaxNode : public Node {
        public:
            /**
             * @brief Constructor to initialize a `SoftmaxNode` for applying the Softmax activation function.
             *
             * The constructor initializes a `SoftmaxNode`, which applies the Softmax activation function to
             * an input tensor. It establishes a connection to the input node, initializes the output tensor,
             * and sets up the node for Softmax computation.
             *
             * @param input A pointer to the input node. Its `output` tensor will have the Softmax activation applied.
             *
             * @details
             * The Softmax activation function is defined as:
             * ```
             * Softmax(x_i) = exp(x_i) / sum(exp(x_j))
             * ```
             * where x_i is the i-th element of the input vector and the sum is over all elements j.
             *
             * Key operations performed by the constructor:
             * - Initializes the `sum` member variable to 0, which may be used in future computations.
             * - Adds the input node to the `inputs` vector, establishing the connection in the computational graph.
             * - Determines if gradient tracking is required based on the input tensor's `requiresGrad` property.
             * - Initializes the `output` tensor with the same shape as the input tensor and appropriate gradient tracking.
             * - Sets the node type to "Softmax" for identification in the computational graph.
             *
             * @note
             * - The Softmax function normalizes the input to a probability distribution over predicted output classes.
             * - This constructor only sets up the node structure; the actual Softmax computation is performed in the forward pass.
             * - Gradient tracking for the output tensor is automatically set based on the input tensor's requirements.
             * - The `sum` variable initialized here may be used for optimizations in the forward or backward passes.
             *
             * @see forward() for the implementation of the Softmax computation in the forward pass.
             * @see backward() for the gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2023/12/06
             */
            explicit SoftmaxNode(Node* input);

            /**
             * @brief Performs the forward pass of the Softmax operation.
             *
             * This method implements the forward computation for the Softmax activation function. It calculates
             * the exponential sum of the input elements and then applies the Softmax function to each element.
             *
             * @details
             * The forward pass is implemented in two main steps:
             * 1. Calculation of the sum of exponentials:
             *    - Uses CUDA parallelization to compute exp(x) for each input element.
             *    - Accumulates these exponentials to get the sum for normalization.
             * 2. Application of the Softmax function:
             *    - Computes exp(x_i) / sum(exp(x_j)) for each element using CUDA.
             *
             * The Softmax function is defined as:
             * ```
             * Softmax(x_i) = exp(x_i) / sum(exp(x_j))
             * ```
             * where x_i is the i-th element of the input vector and the sum is over all elements j.
             *
             * Key operations:
             * - CUDA kernel setup for parallel computation.
             * - Memory allocation and management for intermediate results.
             * - Execution of the SummationExp CUDA kernel for exponential sum calculation.
             * - Data transfer between GPU and CPU for sum accumulation.
             * - Execution of the Softmax CUDA kernel for final output computation.
             *
             * @note
             * - This implementation utilizes CUDA for efficient parallel computation on GPU.
             * - The method handles both the exponential sum calculation and the final Softmax normalization.
             * - Temporary memory is allocated and freed for intermediate calculations.
             * - The final output is stored in the node's output tensor.
             *
             * @see Softmax CUDA kernel for the implementation of the final Softmax computation.
             * @see backward() for the corresponding backward pass implementation.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2023/12/06
             */
            void forward() override;

            /**
             * @brief Performs the backward pass of the Softmax operation.
             *
             * This method implements the gradient computation for the Softmax activation function. It calculates
             * the Jacobian matrix of the Softmax function and then uses it to compute the gradient with respect
             * to the input.
             *
             * @details
             * The backward pass is implemented in two main steps:
             * 1. Calculation of the Softmax Jacobian:
             *    - Computes the Jacobian matrix for the Softmax function using CUDA parallelization.
             * 2. Gradient computation:
             *    - Performs matrix multiplication between the Jacobian and the output gradient to obtain
             *      the input gradient.
             *
             * The Jacobian of the Softmax function is defined as:
             * ```
             * J_ij = softmax_i * (δ_ij - softmax_j)
             * ```
             * where δ_ij is the Kronecker delta.
             *
             * Key operations:
             * - Initialization of the Jacobian tensor.
             * - CUDA kernel setup for parallel computation of the Jacobian.
             * - Execution of the SoftmaxJacobian CUDA kernel to compute the Jacobian matrix.
             * - CUDA kernel setup for matrix multiplication.
             * - Execution of the GeneralMatrixMul CUDA kernel to compute the final gradient.
             *
             * @note
             * - This implementation utilizes CUDA for efficient parallel computation on GPU.
             * - The Jacobian computation and matrix multiplication are performed entirely on the GPU.
             * - The method assumes that the output gradient (output->grad()) has already been set.
             * - The computed gradient is stored in the input node's gradient (inputs[0]->output->grad()).
             *
             * @see forward() for the corresponding forward pass implementation.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2023/12/06
             */
            void backward() override;
        };

        /**
         * @class ReshapeNode
         * @brief Implements tensor shape transformation within a neural network computational graph.
         *
         * The `ReshapeNode` class modifies the dimensional structure of input tensors while preserving
         * their underlying data. This node enables flexible tensor shape adaptation between different
         * network layers without altering the actual data values.
         *
         * @details
         * Core functionality and behavior:
         * - **Shape Transformation**: Reorganizes tensor dimensions according to specified new_shape.
         * - **Data Preservation**: Maintains original data values and memory sharing where possible.
         * - **Zero-Copy Optimization**: Avoids data duplication when compatible memory layouts allow.
         * - **Gradient Propagation**: Correctly routes gradients during backward pass by preserving
         *    original tensor dimensions in gradient computations.
         * - **Runtime Validation**: Verifies shape compatibility (total element count) during forward pass.
         *
         * Implementation specifics:
         * - **Forward Pass**: Adjusts tensor dimensions immediately without data movement.
         * - **Backward Pass**: Restores gradient tensor dimensions to match original input shape.
         * - **Memory Management**: Shares underlying data buffer between input and output tensors.
         * - **CUDA Support**: Maintains device context (CPU/GPU) during reshape operations.
         *
         * Typical applications:
         * - Bridging fully connected layers with convolutional layers requiring specific tensor shapes.
         * - Adapting variable-length sequence inputs to fixed-dimension layer requirements.
         * - Implementing dynamic computational graphs with shape-modifying operations.
         *
         * Critical considerations:
         * - **Element Count Consistency**: Input and new_shape must contain identical total elements.
         * - **Memory Layout Impact**: Reshape success depends on tensor memory continuity in some frameworks.
         * - **Gradient Integrity**: Requires maintaining original input shape reference for correct backpropagation.
         * - **Device Consistency**: Input tensor and new_shape must reside on same computation device.
         *
         * @note
         * - Shape modification doesn't alter tensor data ordering - elements follow memory layout order.
         * - For discontinuous tensors, reshape may trigger implicit data copy (implementation-dependent).
         * - Gradient computation requires preserving original input tensor dimensions throughout node lifetime.
         *
         * @see Node Base class for computational graph nodes
         * @see Tensor::reshape() Underlying tensor shape modification method
         *
         * ### Usage Demonstration:
         * ```cpp
         * // Create input node with 4D tensor
         * InputNode input({2, 3, 4, 5}, true); // Batch 2, 3x4x5 features
         *
         * // Reshape to 2D tensor (batch size, flattened features)
         * ReshapeNode reshape(&input, {2, 1, 1, 3*4*5});
         * reshape.forward();
         *
         * // Verify new shape
         * std::cout << "Reshaped tensor dimensions: "
         *           << reshape.output->shape() << std::endl;
         *
         * // Backward pass demonstration
         * reshape.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2023/10/15
         */
        class DL_API ReshapeNode : public Node {
        public:
            Tensor::shape_type newShape;

            /**
             * @brief Constructs a ReshapeNode object to reshape the input tensor.
             *
             * This constructor initializes a ReshapeNode with an input node and a new shape. It checks if the number of dimensions of the input tensor's shape
             * matches the number of dimensions of the new shape. If they match, it adds the input node to the list of inputs, creates a new output tensor
             * with the specified new shape and the same requiresGrad property as the input tensor, and sets the node type to "Reshape".
             *
             * @param input A pointer to the input node. Memory location: host - to - device (used to access input tensor information).
             * @param newShape A reference to the new shape of type `Tensor::shape_type`. Memory location: host - to - device (used to define the new shape).
             *
             * @return None
             *
             * **Memory Management Strategy**:
             * - The `inputs` vector stores a pointer to the input node. The memory management of the input node is assumed to be handled by the caller.
             * - The `output` member variable is a `std::shared_ptr` to a new `Tensor` object. The memory of the `Tensor` will be automatically managed by the `std::shared_ptr`.
             *
             * **Exception Handling Mechanism**:
             * - Throws `std::invalid_argument` if the number of dimensions of the input tensor's shape does not match the number of dimensions of the new shape.
             *
             * **Relationship with Other Components**:
             * - Depends on the `input` node to access its output tensor.
             * - Creates a new `Tensor` object for the output.
             *
             * @throws std::invalid_argument When the number of dimensions of the input tensor's shape and the new shape do not match.
             *
             * @note
             * - Ensure that the input node is valid and points to a non - null `Node` object.
             * - Ensure that the new shape has the same number of dimensions as the input tensor's shape to avoid exceptions.
             *
             * @code
             * ```cpp
             * Node* inputNode = new Node();
             * Tensor::shape_type newShape = {1, 2, 3, 4};
             * try {
             *     ReshapeNode reshapeNode(inputNode, newShape);
             * } catch (const std::exception& e) {
             *     std::cerr << e.what() << std::endl;
             * }
             * ```
             * @endcode
             */
            ReshapeNode(Node* input, const Tensor::shape_type& newShape);

            /**
             * @brief Performs the forward pass operation of the ReshapeNode.
             *
             * This function copies the data from the output tensor of the input node to the output tensor of the ReshapeNode using CUDA memory copy.
             * It uses the singleton instance of `cuStrm::StreamManager<float>` to manage the CUDA stream for the memory copy operation.
             *
             * @param None
             *
             * @return None
             *
             * **Memory Management Strategy**:
             * - The function relies on the CUDA memory copy operation (`cudaMemcpyDeviceToDevice`). The source and destination memory are managed by the `Tensor` objects (`inputs[0]->output` and `output`).
             * - The CUDA memory copy operation is assumed to handle the actual data transfer and memory allocation/deallocation related to the transfer properly.
             *
             * **Exception Handling Mechanism**:
             * - If the CUDA memory copy operation fails, it may return a CUDA error code. However, this function does not handle CUDA errors explicitly.
             *   Callers should check the CUDA error state after calling this function if necessary.
             *
             * **Relationship with Other Components**:
             * - Depends on the `cuStrm::StreamManager<float>` singleton to manage the CUDA stream for the memory copy.
             * - Relies on the `inputs[0]->output` tensor for the source data and the `output` tensor for the destination data.
             *
             * @throws None explicitly, but CUDA errors may occur during the memory copy operation.
             *
             * @note
             * - Ensure that the CUDA runtime environment is properly initialized before calling this function.
             * - The CUDA memory copy operation assumes that the source and destination memory regions are valid and have sufficient space.
             * - The time complexity of this function is mainly determined by the CUDA memory copy operation, which is typically proportional to the size of the data being copied (O(n), where n is the number of elements in the output tensor).
             *
             * @code
             * ```cpp
             * // Assume ReshapeNode, Node, and Tensor are defined
             * Node* inputNode = new Node();
             * Tensor::shape_type newShape = {1, 2, 3, 4};
             * ReshapeNode reshapeNode(inputNode, newShape);
             * reshapeNode.forward();
             * ```
             * @endcode
             */
            void forward() override;

            /**
             * @brief Performs the backward propagation for the ReshapeNode.
             *
             * @param None
             *
             * @return None
             *
             * This function is responsible for performing the backward propagation in the ReshapeNode.
             * If the output of the first input tensor requires gradient computation, it copies the gradient of the output tensor
             * to the gradient of the first input tensor's output. The memory copy operation is performed using the CUDA stream manager
             * with a device-to-device memory transfer.
             *
             * Memory management strategy: The function does not allocate or free any memory. It only copies existing memory using CUDA's memcpy.
             * Exception handling mechanism: There is no explicit exception handling in this function. However, CUDA's memcpy operation may
             * throw errors if there are issues with the memory pointers or the CUDA device.
             *
             * @note
             * - Ensure that the CUDA device is properly initialized before calling this function.
             * - The sizes of the output tensor and the first input tensor's output must match for the memory copy to be valid.
             *
             * @code
             * ```cpp
             * ReshapeNode node;
             * // Assume inputs and output tensors are properly initialized
             * node.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void backward() override;
        };

        /**
         * @class ExpandNode
         * @brief Expands tensors with batch size 1 to arbitrary batch dimensions through data replication.
         *
         * The `ExpandNode` class enables batch dimension expansion by replicating single-instance input data
         * across the batch dimension. This operation is particularly useful for converting single-sample processing
         * networks into batch-processing configurations without modifying core network architecture.
         *
         * @details
         * Core operational characteristics:
         * - **Batch Dimension Expansion**: Duplicates input data along batch dimension to create specified batch size.
         * - **Memory Efficiency**: Shares underlying data storage through tensor views where possible.
         * - **Gradient Aggregation**: Implements gradient summation across batch dimension during backward pass.
         * - **Device Consistency**: Maintains computation device context (CPU/GPU) during expansion operations.
         * - **Input Validation**: Enforces pre-condition of batch_size=1 for input tensors.
         *
         * Implementation mechanics:
         * - **Forward Propagation**: Creates tensor view with expanded batch dimensions through broadcasting.
         * - **Backward Propagation**: Accumulates gradients across expanded batch dimension via summation.
         * - **CUDA Optimization**: Leverages CUDA tensor broadcasting for efficient batch replication on GPU.
         * - **Shape Preservation**: Maintains non-batch dimensions identical to input tensor.
         *
         * Common application scenarios:
         * - Converting single-sample inference networks to batch processing mode.
         * - Data augmentation through batch-wise replication of prototype samples.
         * - Model ensemble techniques requiring multiple copies of base input.
         *
         * Critical operational constraints:
         * - **Input Batch Requirement**: Input tensor must have batch_size=1 (first dimension size=1).
         * - **Memory Considerations**: Creates virtual views rather than physical copies in forward pass.
         * - **Gradient Scaling**: Backward pass gradients are scaled by batch size due to summation aggregation.
         * - **Device Compatibility**: Input tensor and expansion operation must reside on same computation device.
         *
         * @warning
         * - Input validation failure occurs if input batch size != 1, throwing runtime_error.
         * - Physical memory consumption increases proportionally with new_batch in backward pass.
         *
         * @note
         * - Actual data replication occurs lazily during memory access operations.
         * - For physical data duplication, combine with CopyNode before ExpandNode.
         * - Gradient computation maintains mathematical equivalence to manual batch duplication.
         *
         * @see RepeatNode For non-batch dimension replication operations
         * @see Tensor::expand() Underlying tensor expansion mechanism
         *
         * ### Usage Example:
         * ```cpp
         * // Create single-sample input node
         * InputNode input({1, 1, 256, 256}, true); // Batch 1, 256x256 image
         *
         * // Expand to batch size 16
         * ExpandNode expander(&input, 16);
         * expander.forward();
         *
         * // Verify expanded shape
         * std::cout << "Expanded tensor shape: "
         *           << expander.output->shape() << std::endl; // [16, 1, 256, 256]
         *
         * // Backward pass handling
         * expander.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2023/10/16
         */
        class DL_API ExpandNode : public Node {
        public:
            Tensor::size_type newBatch;

            /**
             * @brief Constructs an ExpandNode object.
             *
             * @param input A pointer to the input Node. Memory location: host.
             * @param newBatch The new batch size for the output tensor. Memory location: host.
             *
             * @return None
             *
             * This function constructs an ExpandNode object. It first checks if the batch size of the input tensor is 1. If not,
             * it throws an `std::invalid_argument` exception. Then, it adds the input node to the list of inputs, creates a new
             * output tensor with the specified new batch size and the same dimensions as the input tensor except for the batch size,
             * and sets the node type to "Expand".
             *
             * Memory management strategy: The function allocates memory for the output tensor using `std::make_shared`. The memory
             * will be automatically managed by the smart pointer and freed when it goes out of scope.
             * Exception handling mechanism: If the batch size of the input tensor is not 1, the function throws an `std::invalid_argument`
             * exception with an appropriate error message.
             *
             * @throws std::invalid_argument If the input tensor's batch size is not 1.
             *
             * @note
             * - Ensure that the input node pointer is valid and points to a properly initialized node.
             * - The new batch size should be a valid non - negative value.
             *
             * @code
             * ```cpp
             * Node* inputNode = new Node();
             * Tensor::size_type newBatchSize = 10;
             * ExpandNode expandNode(inputNode, newBatchSize);
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            ExpandNode(Node* input, Tensor::size_type newBatch);

            /**
             * @brief Performs the forward propagation for the ExpandNode.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the forward propagation of the ExpandNode. It first calculates the size of a single element
             * in the input tensor (excluding the batch dimension) and the total number of elements in the output tensor.
             * Then, it configures the CUDA grid and block dimensions for parallel execution. Finally, it calls the `Expand`
             * kernel function to perform the actual expansion operation on the device.
             *
             * Memory management strategy: The function does not allocate or free any memory directly. It relies on the memory
             * allocated for the input and output tensors in the constructor.
             * Exception handling mechanism: There is no explicit exception handling in this function. However, the `Expand`
             * kernel call may encounter errors related to CUDA operations such as invalid grid/block dimensions or device issues.
             *
             * @throws None
             *
             * @note
             * - Ensure that the CUDA device is properly initialized before calling this function.
             * - The `Expand` kernel function should be implemented correctly to handle the expansion operation.
             * - The time complexity of the expansion operation depends on the implementation of the `Expand` kernel,
             *   but in general, it has a linear time complexity O(n), where n is the total number of elements in the output tensor.
             *
             * @code
             * ```cpp
             * ExpandNode expandNode; // Assume expandNode is properly initialized
             * expandNode.forward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void forward() override;

            /**
             * @brief Performs the backward propagation for the ExpandNode.
             *
             * @param None
             *
             * @return None
             *
             * This function performs the backward propagation of the ExpandNode. It first checks if the input tensor requires
             * gradient computation. If it does, it calculates the size of a single element in the input tensor (excluding the
             * batch dimension) and the total number of elements in the output tensor. Then, it configures the CUDA grid and
             * block dimensions for parallel execution. Finally, it calls the `Compress` kernel function to perform the
             * compression operation on the gradients, which is the reverse operation of the forward expansion.
             *
             * Memory management strategy: The function does not allocate or free any memory directly. It relies on the memory
             * allocated for the input and output gradient tensors.
             * Exception handling mechanism: There is no explicit exception handling in this function. However, the `Compress`
             * kernel call may encounter errors related to CUDA operations such as invalid grid/block dimensions or device issues.
             *
             * @throws None
             *
             * @note
             * - Ensure that the CUDA device is properly initialized before calling this function.
             * - The `Compress` kernel function should be implemented correctly to handle the gradient compression operation.
             * - The time complexity of the compression operation depends on the implementation of the `Compress` kernel,
             *   but in general, it has a linear time complexity O(n), where n is the total number of elements in the output tensor.
             *
             * @code
             * ```cpp
             * ExpandNode expandNode; // Assume expandNode is properly initialized
             * expandNode.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void backward() override;
        };

        /**
         * @class Img2ColNode
         * @brief Implements im2col transformation for efficient convolution operations in neural networks.
         *
         * This node converts (N, C, H, W) input tensors into expanded column matrices (N, 1, Hout*Wout, C*K_h*K_w)
         * following the im2col algorithm, enabling efficient convolution computation through matrix multiplication.
         *
         * @details
         * Core functionality and characteristics:
         * - **Convolution Optimization**: Unfilters image patches into column matrices for accelerated convolution computation.
         * - **Parameterized Transformation**: Supports configurable kernel dimensions, stride, and padding.
         * - **Memory Layout Conversion**: Reorganizes spatial data into channel-patch columns.
         * - **Gradient Propagation**: Implements inverse col2im operation during backward pass for gradient computation.
         * - **CUDA Acceleration**: Optimized GPU implementation for both forward and backward operations.
         * - **Automatic Shape Calculation**: Computes output spatial dimensions based on input parameters.
         *
         * Key implementation aspects:
         * - **Forward Pass**: Applies im2col algorithm with zero-padding and window sliding based on parameters.
         * - **Backward Pass**: Accumulates gradients using col2im inverse transformation with proper kernel alignment.
         * - **Memory Access Patterns**: Optimizes for contiguous memory access in transformed column matrices.
         * - **Parameter Validation**: Verifies kernel dimensions don't exceed padded input size.
         *
         * Typical use cases:
         * - Accelerating convolutional layer computations through matrix multiplication.
         * - Implementing custom filter operations requiring explicit patch extraction.
         * - Converting spatial correlations into channel-wise operations.
         * - Data augmentation through explicit patch sampling.
         *
         * Critical considerations:
         * - **Memory Overhead**: Generates O(K_h*K_w*C*Hout*Wout) intermediate storage - may require substantial memory.
         * - **Parameter Compatibility**: Requires (H + 2*padding - K_h) divisible by stride (similar for width).
         * - **Input Constraints**: Strict 4D input tensor requirement (NCHW format).
         * - **Device Consistency**: Maintains original tensor device context (CPU/GPU).
         *
         * @warning
         * - Excessive kernel sizes or small strides may create prohibitively large intermediate matrices.
         * - Improper padding/stride combinations may cause dimension calculation errors.
         *
         * @note
         * - Output dimensions follow: Hout = (H + 2*padding - K_h)/stride + 1
         * - The inverse col2im operation in backward pass sums gradients across overlapping patches.
         * - For dilated convolutions, consider extending kernel parameters with dilation factors.
         *
         * @see ConvolutionNode For typical subsequent operation after im2col transformation
         * @see Tensor::im2col() Underlying tensor transformation implementation
         *
         * ### Usage Example:
         * ```cpp
         * // Create input node with batch of 32 RGB images (256x256)
         * InputNode input({32, 3, 256, 256}, true);
         *
         * // Configure im2col with 7x7 kernel, stride 2, padding 3
         * Img2ColNode im2col(&input, 7, 7, 2, 3);
         * im2col.forward();
         *
         * // Resulting shape: (32, 1, 128*128, 3*7*7)
         * std::cout << "Transformed shape: " << im2col.output->shape() << std::endl;
         *
         * // Backward pass through col2im
         * im2col.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2023/10/17
         */
        class DL_API Img2ColNode : public Node {
        public:
            Tensor::size_type kernelHeight;
            Tensor::size_type kernelWidth;
            Tensor::size_type stride;
            Tensor::size_type padding;
            Tensor::size_type outputHeight;
            Tensor::size_type outputWidth;

            /**
             * @brief Constructor for the Img2ColNode class.
             *
             * @param input A pointer to the input Node. Memory location: host. This is a pointer to an existing Node object, and the constructor only stores the pointer, not making a copy of the object.
             * @param kernelHeight The height of the kernel. Memory location: host. It is a value passed by value, and the constructor stores its copy.
             * @param kernelWidth The width of the kernel. Memory location: host. It is a value passed by value, and the constructor stores its copy.
             * @param stride The stride value for the convolution operation. Memory location: host. It is a value passed by value, and the constructor stores its copy.
             * @param padding The padding value for the convolution operation. Memory location: host. It is a value passed by value, and the constructor stores its copy.
             *
             * @return None
             *
             * This constructor initializes an Img2ColNode object. It stores the input node pointer, sets the kernel height, width, stride, and padding values. It also calculates the output height and width based on the input tensor's shape, kernel size, stride, and padding. Then, it creates a new Tensor object for the output with the appropriate shape and gradient requirement. Finally, it sets the node type to "Img2Col".
             *
             * Memory management strategy: The constructor creates a new Tensor object using `std::make_shared`, which manages the memory automatically. The input node pointer is just stored, and no new memory is allocated for it.
             * Exception handling mechanism: There is no explicit exception handling in this constructor. However, if the `std::make_shared` call fails to allocate memory for the output Tensor, a `std::bad_alloc` exception will be thrown.
             *
             * @throws std::bad_alloc If memory allocation for the output Tensor fails.
             *
             * @note
             * - Ensure that the input node pointer is valid and points to a properly initialized Node object.
             * - The calculation of output height and width assumes valid values for kernel size, stride, and padding. Incorrect values may lead to unexpected results.
             * - The time complexity of the constructor is O(1) as all operations are constant time operations.
             *
             * @code
             * ```cpp
             * Node* inputNode = new Node();
             * Img2ColNode img2ColNode(inputNode, 3, 3, 1, 1);
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            Img2ColNode(Node* input, Tensor::size_type kernelHeight, Tensor::size_type kernelWidth,
                        Tensor::size_type stride, Tensor::size_type padding);

            /**
             * @brief Performs the forward propagation for the Img2ColNode.
             *
             * @param None
             *
             * @return None
             *
             * This function executes the forward propagation of the Img2ColNode. It calls the `iImg2col` function, passing in the necessary parameters such as the output data pointer, input data pointer, output height, output width, number of input channels, kernel height, kernel width, stride, padding, input height, input width, and batch size. The `iImg2col` function is responsible for converting the image data into a column - major format, which is useful for performing convolution operations more efficiently.
             *
             * Memory management strategy: The function does not allocate or free any memory directly. It relies on the memory already allocated for the input and output tensors. The `iImg2col` function is assumed to write the results directly into the pre - allocated output tensor.
             * Exception handling mechanism: There is no explicit exception handling in this function. However, if the `iImg2col` function encounters issues such as invalid pointers or incorrect input parameters, it may throw an exception.
             *
             * @note
             * - Ensure that the input and output tensors are properly initialized and have sufficient memory allocated before calling this function.
             * - The performance of this function depends on the implementation of the `iImg2col` function. In general, the time complexity of the `iImg2col` operation is O(n), where n is the number of elements in the output tensor.
             *
             * @code
             * ```cpp
             * Img2ColNode img2ColNode; // Assume img2ColNode is properly initialized
             * img2ColNode.forward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void forward() override;

            /**
             * @brief Performs the backward propagation for the Img2ColNode.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the backward propagation of the Img2ColNode. It first checks if the output of the input node requires gradient computation. If so, it calls the `iImg2colBackward` function, passing in relevant parameters including the gradient of the input node's output, the gradient of the current node's output, output height, output width, the number of input channels, kernel height, kernel width, stride, padding, input height, input width, and batch size. The `iImg2colBackward` function is responsible for calculating the gradients with respect to the input.
             *
             * Memory management strategy: The function does not allocate or free any memory directly. It relies on the pre - allocated memory for the gradients of the input and output tensors. The `iImg2colBackward` function is assumed to write the calculated gradients into the pre - allocated gradient tensors.
             * Exception handling mechanism: There is no explicit exception handling in this function. However, if the `iImg2colBackward` function encounters problems such as invalid pointers or incorrect input parameters, it may throw an exception.
             *
             * @note
             * - Ensure that the gradient tensors of the input and output are properly initialized and have sufficient memory allocated before calling this function.
             * - The performance of this function depends on the implementation of the `iImg2colBackward` function. Generally, the time complexity of the `iImg2colBackward` operation is O(n), where n is the number of elements in the input gradient tensor.
             *
             * @code
             * ```cpp
             * Img2ColNode img2ColNode; // Assume img2ColNode is properly initialized
             * img2ColNode.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void backward() override;
        };

        /**
         * @class Col2ImgNode
         * @brief Reconstructs spatial tensors from column matrices generated by im2col transformation.
         *
         * This node performs the inverse operation of Img2ColNode, converting (N, 1, Hout*Wout, Cout) tensors
         * back to (N, Cout, Hout, Wout) spatial format. Essential for gradient propagation in convolution
         * operations and feature map reconstruction.
         *
         * @details
         * Core functionality and characteristics:
         * - **Spatial Reconstruction**: Transforms column-organized data back to original spatial dimensions.
         * - **Gradient Handling**: Reconstructs gradients during backpropagation through reverse im2col.
         * - **Parameter Consistency**: Requires matching kernel/stride/padding with original Img2ColNode.
         * - **Memory Optimization**: Implements efficient gradient summation for overlapping regions.
         * - **Shape Restoration**: Recovers original spatial dimensions through inverse indexing.
         * - **CUDA Support**: GPU-accelerated implementation for both forward and backward passes.
         *
         * Key implementation aspects:
         * - **Forward Pass**: Reshapes column matrix to spatial format using output dimensions.
         * - **Backward Pass**: Applies im2col transformation to gradients for upstream propagation.
         * - **Overlap Resolution**: Accumulates gradients from overlapping input regions during backward pass.
         * - **Shape Validation**: Verifies input tensor conforms to (N, 1, H*W, C) structure.
         *
         * Typical use cases:
         * - Gradient computation for convolution layers using matrix-based implementations.
         * - Feature map reconstruction after channel-wise operations.
         * - Custom layer implementations requiring spatial/tensor format conversion.
         * - Visualization of intermediate column-organized features.
         *
         * Critical considerations:
         * - **Parameter Matching**: Must use identical kernel/stride/padding to corresponding Img2ColNode.
         * - **Memory Footprint**: Gradient tensors may require significant memory during backward pass.
         * - **Input Constraints**: Strict 4D input tensor requirement with second dimension = 1.
         * - **Spatial Resolution**: Output dimensions must match original pre-im2col calculations.
         *
         * @warning
         * - Input tensor shape mismatch will cause runtime errors.
         * - Incorrect kernel/stride parameters may produce distorted output spatial dimensions.
         * - Large kernel sizes combined with small strides may create memory pressure during backward pass.
         *
         * @note
         * - Output dimensions should satisfy: Hout = (H - K_h + 2*padding)/stride + 1
         * - Backward pass internally uses Img2ColNode's logic for gradient propagation.
         * - For dilated convolutions, ensure parameter compatibility with forward operations.
         *
         * @see Img2ColNode For the corresponding forward transformation
         * @see ConvolutionNode For typical usage context in convolutional networks
         *
         * ### Usage Example:
         * ```cpp
         * // Previous im2col transformation
         * Img2ColNode im2col(&input, 3, 3, 1, 1);
         * im2col.forward(); // Output shape: (32, 1, 256*256, 64)
         *
         * // Reconstruct original spatial format
         * Col2ImNode col2im(&im2col, 3, 3, 1, 1);
         * col2im.forward();
         *
         * // Reconstructed shape: (32, 64, 256, 256)
         * std::cout << "Reconstructed shape: " << col2im.output->shape() << std::endl;
         *
         * // Backward pass through inverse transformation
         * col2im.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2023/10/18
         */
        class DL_API Col2ImgNode : public Node {
        public:
            Tensor::size_type outputHeight;
            Tensor::size_type outputWidth;
            Tensor::size_type outputChannels;

            /**
             * @brief Constructor for the Col2ImgNode class.
             *
             * @param input A pointer to the input Node (host-to-object). This node provides the necessary input data for the Col2ImgNode.
             * @param outputHeight The height of the output image in size_type (host-to-object).
             * @param outputWidth The width of the output image in size_type (host-to-object).
             *
             * @return None
             *
             * This constructor initializes a Col2ImgNode object. It sets the output height, width, and number of channels based on the input node's output shape. It then adds the input node to the list of input nodes. A new Tensor object is created for the output, with its shape determined by the batch size from the input node's output, the number of output channels, the specified output height, and width. The output tensor also inherits the gradient requirement from the input node's output. Finally, the node type is set to "Col2Img".
             *
             * Memory management strategy: The constructor uses `std::make_shared` to allocate memory for the output tensor. The memory will be automatically managed by the smart pointer, and it will be released when the last reference to the tensor is removed.
             * Exception handling mechanism: There is no explicit exception handling in this constructor. However, if memory allocation for the output tensor fails, a `std::bad_alloc` exception may be thrown.
             *
             * @throws std::bad_alloc If memory allocation for the output tensor fails.
             *
             * @note
             * - Ensure that the input pointer is valid and points to a properly initialized Node object.
             * - The performance of this constructor is mainly determined by the memory allocation for the output tensor, which has a time complexity of O(n), where n is the number of elements in the output tensor.
             *
             * @code
             * ```cpp
             * Node* inputNode = new Node();
             * Tensor::size_type outputHeight = 10;
             * Tensor::size_type outputWidth = 10;
             * Col2ImgNode col2ImgNode(inputNode, outputHeight, outputWidth);
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            Col2ImgNode(Node* input, Tensor::size_type outputHeight, Tensor::size_type outputWidth);

            /**
             * @brief Performs the forward propagation operation in the Col2ImgNode.
             *
             * @param None
             *
             * @return None
             *
             * This function executes the forward propagation step for the Col2ImgNode. It calls the `iCol2img` function, passing the output tensor's data, the input node's output tensor's data, the output height, output width, output channels, and the batch size from the input node's output. The `iCol2img` function is responsible for converting the column - formatted data from the input to image - formatted data in the output.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing data pointers of the input and output tensors.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `iCol2img` function encounters an error, it may throw an exception, but the nature of the exception depends on the implementation of `iCol2img`.
             *
             * @throws [Exception type from iCol2img] If the `iCol2img` function encounters an error during execution.
             *
             * @note
             * - Ensure that the input node's output tensor and the output tensor of the Col2ImgNode are properly initialized before calling this function.
             * - The performance of this function depends on the implementation of the `iCol2img` function. If the `iCol2img` function has a time complexity of O(n), where n is the number of elements in the input or output tensors, then this forward propagation step also has a time complexity of O(n).
             *
             * @code
             * ```cpp
             * Col2ImgNode col2ImgNode(...); // Assume Col2ImgNode is properly initialized
             * col2ImgNode.forward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void forward() override;

            /**
             * @brief Performs the backward propagation operation in the Col2ImgNode.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the backward propagation step for the Col2ImgNode. It first checks if the output tensor of the input node requires gradient computation. If it does, the function calls `iCol2imgBackward`, passing the gradient tensor of the input node's output, the gradient tensor of the output, the output height, output width, output channels, and the batch size from the input node's output. The `iCol2imgBackward` function is responsible for computing the gradients and propagating them back to the input.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing gradient tensors of the input and output.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `iCol2imgBackward` function encounters an error, it may throw an exception, and the specific type of exception depends on the implementation of `iCol2imgBackward`.
             *
             * @throws [Exception type from iCol2imgBackward] If the `iCol2imgBackward` function encounters an error during execution.
             *
             * @note
             * - Ensure that the gradient tensors of the input node's output and the output tensor of the Col2ImgNode are properly initialized before calling this function.
             * - The performance of this function depends on the implementation of the `iCol2imgBackward` function. If the `iCol2imgBackward` function has a time complexity of O(n), where n is the number of elements in the input or output gradient tensors, then this backward propagation step also has a time complexity of O(n).
             *
             * @code
             * ```cpp
             * Col2ImgNode col2ImgNode(...); // Assume Col2ImgNode is properly initialized
             * col2ImgNode.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void backward() override;
        };

        /**
         * @class AveragePoolingNode
         * @brief Implements average pooling operation for spatial downsampling in neural networks.
         *
         * This node performs spatial averaging over sliding windows of (poolSize x poolSize) dimensions,
         * reducing feature map resolution while maintaining channel depth. Commonly used for dimensionality
         * reduction and translation invariance in CNNs.
         *
         * @details
         * Core functionality and characteristics:
         * - **Spatial Aggregation**: Computes mean values over local receptive fields.
         * - **Dimensionality Reduction**: Reduces H and W dimensions while preserving channel count.
         * - **Gradient Propagation**: Distributes gradients equally across input positions during backward pass.
         * - **Parameter Control**: Configurable window size, stride, and padding.
         * - **CUDA Acceleration**: GPU-optimized implementation for both forward and backward passes.
         * - **Auto Shape Calculation**: Computes output dimensions using pooling parameters.
         *
         * Key implementation aspects:
         * - **Forward Pass**: Applies sliding window averaging with optional zero-padding.
         * - **Backward Pass**: Scatters gradients by assigning 1/(K*K) of output gradient to each input position.
         * - **Border Handling**: Supports padding for controlling output size.
         * - **Memory Efficiency**: Maintains original channel depth for memory layout consistency.
         *
         * Typical use cases:
         * - Reducing computational complexity in deep networks.
         * - Creating translation-invariant feature representations.
         * - Smoothing feature responses in encoder-decoder architectures.
         * - Regularization through spatial information compression.
         *
         * Critical considerations:
         * - **Information Loss**: Over-aggressive pooling may discard important spatial details.
         * - **Dimension Calculation**: Output dimensions follow: Hout = (H + 2*padding - poolSize)/stride + 1
         * - **Parameter Validation**: Requires (H + 2*padding) ≥ poolSize and similar for width.
         * - **Zero Padding Impact**: Padding values affect border region averages.
         *
         * @warning
         * - Excessive pooling sizes may produce invalid dimensions.
         * - Small stride values can significantly increase computation cost.
         * - Non-integer output dimensions will cause runtime errors.
         *
         * @note
         * - Output tensor shape: (N, C, Hout, Wout)
         * - For variable height/width pooling, use separate parameters (not supported in current implementation)
         * - Gradient computation requires storing input shape during forward pass
         *
         * @see MaxPoolingNode For maximum value-based pooling alternative
         *
         * ### Usage Example:
         * ```cpp
         * // Create input node with batch of 16 256x256 RGB images
         * InputNode input({16, 3, 256, 256}, true);
         *
         * // Configure 4x4 pooling with stride 2 and padding 1
         * AveragePoolingNode pool(&input, 4, 2, 1);
         * pool.forward();
         *
         * // Output shape becomes (16, 3, 128, 128)
         * std::cout << "Pooled shape: " << pool.output->shape() << std::endl;
         *
         * // Backpropagate through pooling operation
         * pool.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2023/10/19
         */
        class DL_API AveragePoolingNode : public Node {
        public:
            Tensor::size_type poolSize;
            Tensor::size_type stride;
            Tensor::size_type padding;

            /**
             * @brief Constructs an AveragePoolingNode object.
             *
             * @param input A pointer to the input node. The memory of this pointer is assumed to be managed externally and is used in a read - only manner within this constructor (host - to - host).
             * @param poolSize The size of the pooling window. It is a value of type Tensor::size_type and is used to determine the dimensions of the pooling operation.
             * @param stride The stride value for the pooling operation. It is of type Tensor::size_type and controls how the pooling window moves across the input tensor.
             * @param padding The padding value applied to the input tensor before the pooling operation. It is of type Tensor::size_type.
             *
             * @return None
             *
             * This constructor initializes an AveragePoolingNode object. It first stores the provided input node pointer in the `inputs` vector. Then, it creates a new shared pointer to a Tensor object for the `output` member. The shape of the output tensor is calculated based on the shape of the input tensor, the `poolSize`, `stride`, and `padding` values using the `OUTPUT_DIM` macro. The `requiresGrad` flag of the output tensor is set to the same value as that of the input tensor's output. Finally, it sets the `type` member of the node to "AveragePooling".
             *
             * Memory management strategy: The constructor does not allocate memory for the input node. It only stores a pointer to it. The output tensor is created using `std::make_shared`, which manages the memory automatically.
             * Exception handling mechanism: There is no explicit exception handling in this constructor. If the `std::make_shared` call fails to allocate memory for the output tensor, it may throw a `std::bad_alloc` exception.
             *
             * @throws std::bad_alloc If memory allocation for the output tensor fails.
             *
             * @note
             * - Ensure that the input node pointer is valid and points to a properly initialized node.
             * - The performance of this constructor is mainly determined by the memory allocation for the output tensor, which has a time complexity of O(1) for the pointer management and O(m) for the tensor data allocation, where m is the number of elements in the output tensor.
             *
             * @code
             * ```cpp
             * Node* inputNode = new Node(...);
             * Tensor::size_type poolSize = 2;
             * Tensor::size_type stride = 2;
             * Tensor::size_type padding = 0;
             * AveragePoolingNode avgPoolingNode(inputNode, poolSize, stride, padding);
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            AveragePoolingNode(Node* input, Tensor::size_type poolSize, Tensor::size_type stride,
                               Tensor::size_type padding);

            /**
             * @brief Performs the backward pass of the average pooling operation.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the backward pass of the average pooling operation. It first checks if the output tensor of the input node requires gradient computation. If it does, the function calls `iAveragePoolingBackward`, passing the gradient tensor of the input node's output, the gradient tensor of the output, the pooling size, stride, padding, and the dimensions of the input and output tensors. The `iAveragePoolingBackward` function computes the gradients and propagates them back to the input.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing gradient tensors of the input and output.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `iAveragePoolingBackward` function encounters an error, it may throw an exception, and the specific type of exception depends on the implementation of `iAveragePoolingBackward`.
             *
             * @throws [Exception type from iAveragePoolingBackward] If the `iAveragePoolingBackward` function encounters an error during execution.
             *
             * @note
             * - Ensure that the gradient tensors of the input node's output and the output tensor of the AveragePoolingNode are properly initialized before calling this function.
             * - The performance of this function depends on the implementation of the `iAveragePoolingBackward` function. If the `iAveragePoolingBackward` function has a time complexity of O(n), where n is the number of elements in the input or output gradient tensors, then this backward pass also has a time complexity of O(n).
             *
             * @code
             * ```cpp
             * AveragePoolingNode avgPoolingNode(...); // Assume AveragePoolingNode is properly initialized
             * avgPoolingNode.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void forward() override;

            /**
             * @brief Performs the backward pass of the average pooling operation.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the backward pass of the average pooling operation. It first checks if the output tensor of the input node requires gradient computation. If it does, the function calls `iAveragePoolingBackward`, passing the gradient tensor of the input node's output, the gradient tensor of the output, the pooling size, stride, padding, and the dimensions of the input and output tensors. The `iAveragePoolingBackward` function computes the gradients and propagates them back to the input.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing gradient tensors of the input and output.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `iAveragePoolingBackward` function encounters an error, it may throw an exception, and the specific type of exception depends on the implementation of `iAveragePoolingBackward`.
             *
             * @throws [Exception type from iAveragePoolingBackward] If the `iAveragePoolingBackward` function encounters an error during execution.
             *
             * @note
             * - Ensure that the gradient tensors of the input node's output and the output tensor of the AveragePoolingNode are properly initialized before calling this function.
             * - The performance of this function depends on the implementation of the `iAveragePoolingBackward` function. If the `iAveragePoolingBackward` function has a time complexity of O(n), where n is the number of elements in the input or output gradient tensors, then this backward pass also has a time complexity of O(n).
             *
             * @code
             * ```cpp
             * AveragePoolingNode avgPoolingNode(...); // Assume AveragePoolingNode is properly initialized
             * avgPoolingNode.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void backward() override;
        };

        /**
         * @class GlobalAvgPoolNode
         * @brief Performs global average pooling operation across spatial dimensions of input tensor.
         *
         * This node reduces each channel's spatial dimensions (H, W) to a single average value,
         * producing output of shape (N, C, 1, 1). Commonly used for final feature aggregation before
         * fully connected layers in CNN architectures.
         *
         * @details
         * Core functionality and characteristics:
         * - **Global Aggregation**: Computes channel-wise mean over entire spatial dimensions.
         * - **Dimensionality Collapse**: Reduces H and W dimensions to 1 while preserving channels.
         * - **Gradient Distribution**: Evenly distributes gradients across all spatial positions during backward pass.
         * - **Parameter-Free**: No learnable parameters required for operation.
         * - **CUDA Support**: Optimized GPU implementation for both forward and backward passes.
         * - **Input Preservation**: Maintains batch and channel dimensions unchanged.
         *
         * Key implementation aspects:
         * - **Forward Pass**: Computes spatial average for each channel using tensor reduction.
         * - **Backward Pass**: Scatters gradients by assigning 1/(H*W) of output gradient to each input position.
         * - **Shape Handling**: Automatically handles varying input sizes through dynamic shape inference.
         * - **Memory Efficiency**: Significantly reduces memory footprint for downstream operations.
         *
         * Typical use cases:
         * - Final spatial information aggregation before classification layers.
         * - Network-in-network architectures requiring compact feature representation.
         * - Squeeze-and-Excitation modules for channel-wise attention.
         * - Reducing model parameters in transition layers.
         *
         * Critical considerations:
         * - **Irreversible Compression**: Destroys all spatial information in input features.
         * - **Input Requirements**: Expects 4D input tensor (N, C, H, W).
         * - **Normalization Factor**: Uses 1/(H*W) for both forward average and backward gradient scaling.
         *
         * @warning
         * - Input tensor must have spatial dimensions (H, W) > 0.
         * - Not suitable for preserving spatial relationships in features.
         * - May produce non-sensical gradients if applied to single-element spatial dimensions.
         *
         * @note
         * - Output tensor shape: (N, C, 1, 1)
         * - Often used as alternative to flattening+fully connected layers
         * - Particularly effective in combination with convolutional bottlenecks
         *
         * @see AveragePoolingNode For local spatial averaging with configurable windows
         * @see FlattenNode For alternative spatial-to-vector conversion
         *
         * ### Usage Example:
         * ```cpp
         * // Input features from convolutional backbone
         * InputNode input({16, 512, 7, 7}, true);
         *
         * // Apply global average pooling
         * GlobalAvgPoolNode gap(&input);
         * gap.forward();
         *
         * // Output shape becomes (16, 512, 1, 1)
         * std::cout << "Pooled shape: " << gap.output->shape() << std::endl;
         *
         * // Backpropagate through global averaging
         * gap.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2023/10/20
         */
        class DL_API GlobalAvgPoolNode : public Node {
        public:
            /**
             * @brief Constructs a GlobalAvgPoolNode object.
             *
             * @param input A pointer to the input node. The memory of this pointer is assumed to be managed externally and is used in a read - only manner within this constructor (host - to - host).
             *
             * @return None
             *
             * This constructor initializes a GlobalAvgPoolNode object. It first adds the provided input node pointer to the `inputs` vector. Then, it creates a new shared pointer to a Tensor object for the `output` member. The shape of the output tensor is set to have the same batch size and number of channels as the input tensor's output, but with a height and width of 1. The `requiresGrad` flag of the output tensor is set to the same value as that of the input tensor's output. Finally, it sets the `type` member of the node to "GlobalAvgPool".
             *
             * Memory management strategy: The constructor does not allocate memory for the input node. It only stores a pointer to it. The output tensor is created using `std::make_shared`, which manages the memory automatically.
             * Exception handling mechanism: There is no explicit exception handling in this constructor. If the `std::make_shared` call fails to allocate memory for the output tensor, it may throw a `std::bad_alloc` exception.
             *
             * @throws std::bad_alloc If memory allocation for the output tensor fails.
             *
             * @note
             * - Ensure that the input node pointer is valid and points to a properly initialized node.
             * - The performance of this constructor is mainly determined by the memory allocation for the output tensor, which has a time complexity of O(1) for the pointer management and O(m) for the tensor data allocation, where m is the number of elements in the output tensor (equal to the batch size times the number of channels in this case).
             *
             * @code
             * ```cpp
             * Node* inputNode = new Node(...);
             * GlobalAvgPoolNode globalAvgPoolNode(inputNode);
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            GlobalAvgPoolNode(Node* input);

            /**
             * @brief Performs the forward pass of the global average pooling operation.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the forward pass of the global average pooling operation. It iterates over each sample in the batch (`i` loop) and each channel (`j` loop) of the input tensor. For each combination of batch and channel, it calculates the sum of all elements in the corresponding 2D matrix of the input tensor using the `sum` method. Then, it divides this sum by the total number of elements in the 2D matrix (which is the product of the height and width of the input tensor). The result is then used to fill the corresponding element in the output tensor using the `fillMatrix` method.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing data tensors of the input and output.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `sum` or `fillMatrix` methods encounter an error, they may throw exceptions depending on their implementation.
             *
             * @throws [Exception type from sum or fillMatrix] If the `sum` or `fillMatrix` methods encounter an error during execution.
             *
             * @note
             * - Ensure that the input and output tensors are properly initialized before calling this function.
             * - The time complexity of this function is O(b * c * h * w), where b is the batch size, c is the number of channels, h is the height, and w is the width of the input tensor, because it needs to sum all elements in each 2D matrix of the input tensor.
             *
             * @code
             * ```cpp
             * GlobalAvgPoolNode globalAvgPoolNode(...); // Assume GlobalAvgPoolNode is properly initialized
             * globalAvgPoolNode.forward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void forward() override;

            /**
             * @brief Performs the backward pass of the global average pooling operation.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the backward pass of the global average pooling operation. It first checks if the output tensor of the input node requires gradient computation. If it does, the function calls `iGlobalAvgPoolBackward`, passing the gradient tensor of the input node's output, the gradient tensor of the output, and the shape information of the input tensor (batch size, number of channels, height, and width). The `iGlobalAvgPoolBackward` function computes the gradients and propagates them back to the input.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing gradient tensors of the input and output.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `iGlobalAvgPoolBackward` function encounters an error, it may throw an exception, and the specific type of exception depends on the implementation of `iGlobalAvgPoolBackward`.
             *
             * @throws [Exception type from iGlobalAvgPoolBackward] If the `iGlobalAvgPoolBackward` function encounters an error during execution.
             *
             * @note
             * - Ensure that the gradient tensors of the input node's output and the output tensor of the GlobalAvgPoolNode are properly initialized before calling this function.
             *
             * @code
             * ```cpp
             * GlobalAvgPoolNode globalAvgPoolNode(...); // Assume GlobalAvgPoolNode is properly initialized
             * globalAvgPoolNode.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void backward() override;
        };

        /**
         * @class MaxPoolingNode
         * @brief Implements max pooling operation for spatial downsampling with feature preservation.
         *
         * This node performs spatial max selection over sliding windows of (poolSize x poolSize) dimensions,
         * maintaining the most prominent features while reducing spatial resolution. Essential for preserving
         * sharp feature responses in CNNs.
         *
         * @details
         * Core functionality and characteristics:
         * - **Feature Selection**: Captures maximum activation values within local receptive fields.
         * - **Position Tracking**: Records max value locations via position tensor for gradient computation.
         * - **Sparse Gradient Flow**: Propagates gradients only to winning positions during backward pass.
         * - **Dimensionality Control**: Reduces H/W dimensions while preserving channel structure.
         * - **Border Handling**: Supports configurable padding for output size management.
         * - **CUDA Optimization**: Accelerated implementations for both forward and backward operations.
         * - **Dynamic Shape Adaption**: Automatically calculates output dimensions using pooling parameters.
         *
         * Key implementation aspects:
         * - **Forward Pass**: Computes max values and stores positions using parallelized argmax operations.
         * - **Backward Pass**: Scatters gradients exclusively to original max positions with 1:1 mapping.
         * - **Memory Tradeoff**: Stores position indices (poolSize^2 elements per window) for gradient accuracy.
         * - **Numerical Stability**: Handles multiple equal maxima by selecting first occurrence.
         *
         * Typical use cases:
         * - Dominant feature extraction in early CNN layers
         * - Downsampling while preserving edge/texture information
         * - Constructing translation-invariant representations
         * - Reducing computational load in deep networks
         *
         * Critical considerations:
         * - **Information Discard**: Non-maximal values are irreversibly discarded during forward pass
         * - **Dimension Constraints**: Output dimensions must satisfy:
         *   Hout = (H + 2*padding - poolSize)/stride + 1
         * - **Position Storage**: Requires O(N*C*Hout*Wout) memory for position tracking
         *
         * @warning
         * - Invalid dimension combinations will throw runtime errors
         * - Overly aggressive pooling (poolSize >> stride) may cause feature loss
         * - Position tensor initialization must match forward pass execution order
         *
         * @note
         * - Output shape: (N, C, Hout, Wout)
         * - For variable pooling windows, extend with additional size parameters
         * - Compared to average pooling, better preserves feature sharpness
         *
         * @see AveragePoolingNode For smoothing-based spatial aggregation
         * @see GlobalAvgPoolNode For complete spatial dimension collapse
         *
         * ### Usage Example:
         * ```cpp
         * // Process 32 samples of 128x128 RGB images
         * InputNode input({32, 3, 128, 128}, true);
         *
         * // Create 3x3 max pooling with stride 2 and padding 1
         * MaxPoolingNode mpool(&input, 3, 2, 1);
         * mpool.forward();
         *
         * // Resulting shape: (32, 3, 64, 64)
         * std::cout << "Pooled shape: " << mpool.output->shape() << std::endl;
         *
         * // Backpropagate gradients through max positions
         * mpool.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2023/10/21
         */
        class DL_API MaxPoolingNode : public Node {
        public:
            std::shared_ptr<Tensor> position;
            Tensor::size_type poolSize;
            Tensor::size_type stride;
            Tensor::size_type padding;

            /**
             * @brief Constructs a MaxPoolingNode object.
             *
             * @param input A pointer to the input node. Memory of this pointer is assumed to be managed externally and used in a read - only way within the constructor (host - to - host).
             * @param poolSize The size of the pooling window, of type `Tensor::size_type`. It is passed by value.
             * @param stride The stride of the pooling operation, of type `Tensor::size_type`. It is passed by value.
             * @param padding The padding applied to the input tensor, of type `Tensor::size_type`. It is passed by value.
             *
             * @return None
             *
             * This constructor initializes a MaxPoolingNode object. It first stores the provided `poolSize`, `stride`, and `padding` values. Then, it adds the input node pointer to the `inputs` vector. Next, it creates a new shared pointer to a Tensor object for the `output` member. The shape of the output tensor is calculated based on the input tensor's shape, `poolSize`, `stride`, and `padding` using the `OUTPUT_DIM` macro. The `requiresGrad` flag of the output tensor is set to the same value as that of the input tensor's output. Additionally, it creates a new shared pointer to a Tensor object for the `position` member. The shape of the `position` tensor is the same as that of the output tensor, and its `requiresGrad` flag is set to `false`.
             *
             * Memory management strategy: The constructor does not allocate memory for the input node. It only stores a pointer to it. The `output` and `position` tensors are created using `std::make_shared`, which manages their memory automatically.
             * Exception handling mechanism: There is no explicit exception handling in this constructor. If the `std::make_shared` calls fail to allocate memory for the `output` or `position` tensors, they may throw a `std::bad_alloc` exception.
             *
             * @throws std::bad_alloc If memory allocation for the `output` or `position` tensors fails.
             *
             * @note
             * - Ensure that the input node pointer is valid and points to a properly initialized node.
             * - The performance of this constructor is mainly determined by the memory allocation for the `output` and `position` tensors, which has a time complexity of O(1) for the pointer management and O(m) for the tensor data allocation, where m is the number of elements in the tensors.
             *
             * @code
             * ```cpp
             * Node* inputNode = new Node(...); // Assume Node is properly initialized
             * MaxPoolingNode maxPoolingNode(inputNode, 2, 2, 0);
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            MaxPoolingNode(Node* input, Tensor::size_type poolSize, Tensor::size_type stride,
                           Tensor::size_type padding);

            /**
             * @brief Performs the forward pass of the max - pooling operation.
             *
             * @param None
             *
             * @return None
             *
             * This function executes the forward pass of the max - pooling operation. It calls the `iMaxPooling` function, passing the data pointers of the output tensor, the position tensor, and the input tensor's output. It also provides the pooling parameters (`poolSize`, `stride`, `padding`) and the shape information of the input and output tensors. The `iMaxPooling` function is responsible for computing the max - pooling result and storing it in the output tensor, as well as recording the positions of the maximum values in the position tensor.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing data tensors of the input, output, and position.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `iMaxPooling` function encounters an error, it may throw an exception depending on its implementation.
             *
             * @throws [Exception type from iMaxPooling] If the `iMaxPooling` function encounters an error during execution.
             *
             * @note
             * - Ensure that the input, output, and position tensors are properly initialized before calling this function.
             * - The performance of this function depends on the implementation of the `iMaxPooling` function. The time complexity typically depends on the size of the input tensor and the pooling parameters, and may be O(b * c * h * w) in the worst - case scenario, where b is the batch size, c is the number of channels, h is the height, and w is the width of the input tensor.
             *
             * @code
             * ```cpp
             * MaxPoolingNode maxPoolingNode(...); // Assume MaxPoolingNode is properly initialized
             * maxPoolingNode.forward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void forward() override;

            /**
             * @brief Performs the backward pass of the max - pooling operation.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the backward pass of the max - pooling operation. It first checks if the output tensor of the input node requires gradient computation. If it does, it calls the `iMaxPoolingBackward` function, passing the gradient pointer of the input node's output, the data pointer of the position tensor, the gradient pointer of the output tensor, the pooling parameters (`poolSize`, `stride`, `padding`), and the shape information of the input and output tensors. The `iMaxPoolingBackward` function is responsible for computing the gradients and propagating them back to the input tensor.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing gradient tensors of the input and output, as well as the data tensor of the position.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `iMaxPoolingBackward` function encounters an error, it may throw an exception depending on its implementation.
             *
             * @throws [Exception type from iMaxPoolingBackward] If the `iMaxPoolingBackward` function encounters an error during execution.
             *
             * @note
             * - Ensure that the input, output, and position tensors' gradient and data are properly initialized before calling this function.
             * - The performance of this function depends on the implementation of the `iMaxPoolingBackward` function. The time complexity typically depends on the size of the input tensor and the pooling parameters, and may be O(b * c * h * w) in the worst - case scenario, where b is the batch size, c is the number of channels, h is the height, and w is the width of the input tensor.
             *
             * @code
             * ```cpp
             * MaxPoolingNode maxPoolingNode(...); // Assume MaxPoolingNode is properly initialized
             * maxPoolingNode.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void backward() override;
        };

        /**
         * @class GlobalMaxPoolNode
         * @brief Performs global max pooling operation across spatial dimensions of input tensor.
         *
         * This node reduces each channel's spatial dimensions (H, W) to a single maximum value,
         * producing output of shape (N, C, 1, 1). Used to extract the most salient spatial features
         * while maintaining channel-wise information.
         *
         * @details
         * Core functionality and characteristics:
         * - **Global Feature Selection**: Captures maximum activation per channel across all spatial positions.
         * - **Position Tracking**: Implicitly records max value locations for gradient routing.
         * - **Sparse Gradient Flow**: Propagates gradients exclusively to original max positions during backward pass.
         * - **Dimensionality Collapse**: Reduces H and W dimensions to 1 while preserving channels.
         * - **Parameter-Free**: No learnable parameters required for operation.
         * - **CUDA Support**: Optimized GPU implementation for both forward and backward passes.
         *
         * Key implementation aspects:
         * - **Forward Pass**: Computes channel-wise maxima using tensor reduction with argmax tracking.
         * - **Backward Pass**: Scatters full output gradient to corresponding max positions in input tensor.
         * - **Efficient Storage**: Requires O(N*C) memory for position indices compared to spatial dimensions.
         * - **Tie Resolution**: Selects first occurrence of maximum value when multiple maxima exist.
         *
         * Typical use cases:
         * - Highlighting strongest activation patterns in feature maps
         * - Final feature aggregation for classification tasks
         * - Networks requiring spatial position invariance
         * - Attention mechanisms focusing on dominant features
         *
         * Critical considerations:
         * - **Feature Discard**: Discards all non-maximal spatial information
         * - **Gradient Sparsity**: Only single position per channel receives gradients
         * - **Input Requirements**: Expects 4D input tensor (N, C, H, W) with H, W > 0
         *
         * @warning
         * - Input tensor must have spatial dimensions (H, W) > 1 to be meaningful
         * - Not suitable for tasks requiring spatial relationship preservation
         * - May produce unstable gradients if input contains multiple equal maxima
         *
         * @note
         * - Output tensor shape: (N, C, 1, 1)
         * - Often used as alternative to global average pooling for sharper feature selection
         * - Common in architectures emphasizing dominant visual patterns
         *
         * @see GlobalAvgPoolNode For smoothed spatial feature aggregation
         * @see MaxPoolingNode For local window-based max pooling
         *
         * ### Usage Example:
         * ```cpp
         * // Process batch of 16 samples with 512 channels (7x7 spatial)
         * InputNode input({16, 512, 7, 7}, true);
         * 
         * // Apply global max pooling
         * GlobalMaxPoolNode gmp(&input);
         * gmp.forward();
         * 
         * // Output shape becomes (16, 512, 1, 1)
         * std::cout << "Pooled shape: " << gmp.output->shape() << std::endl;
         *
         * // Backpropagate through max positions
         * gmp.backward();
         * ```
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2023/10/22
         */
        class DL_API GlobalMaxPoolNode : public Node {
        public:
            /**
             * @brief Constructs a GlobalMaxPoolNode object.
             *
             * @param input A pointer to the input node. This pointer is assumed to be managed externally, and the constructor uses it in a read - only manner (host - to - host).
             *
             * @return None
             *
             * This constructor initializes a GlobalMaxPoolNode object. It first adds the provided input node pointer to the `inputs` vector. Then, it creates a new shared pointer for the `output` member. The shape of the output tensor is set to have the same batch size and number of channels as the input tensor's output, but with a height and width of 1. The `requiresGrad` flag of the output tensor is set to the same value as that of the input tensor's output. Finally, it sets the `type` member to the string "GlobalMaxPool".
             *
             * Memory management strategy: The constructor does not allocate memory for the input node. It only stores a pointer to it. The `output` tensor is created using `std::make_shared`, which manages its memory automatically.
             * Exception handling mechanism: There is no explicit exception handling in this constructor. If the `std::make_shared` call fails to allocate memory for the `output` tensor, it may throw a `std::bad_alloc` exception.
             *
             * @throws std::bad_alloc If memory allocation for the `output` tensor fails.
             *
             * @note
             * - Ensure that the input node pointer is valid and points to a properly initialized node.
             * - The performance of this constructor is mainly determined by the memory allocation for the `output` tensor, which has a time complexity of O(1) for the pointer management and O(m) for the tensor data allocation, where m is the number of elements in the output tensor.
             *
             * @code
             * ```cpp
             * Node* inputNode = new Node(...); // Assume Node is properly initialized
             * GlobalMaxPoolNode globalMaxPoolNode(inputNode);
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            GlobalMaxPoolNode(Node* input);

            /**
             * @brief Performs the forward pass of the global max - pooling operation.
             *
             * @param None
             *
             * @return None
             *
             * This function conducts the forward pass of the global max - pooling operation. It iterates over each batch and channel of the input tensor's output. For each combination of batch index `i` and channel index `j`, it computes the maximum value in the corresponding slice of the input tensor using the `max` method. Then, it fills the corresponding position in the output tensor using the `fillMatrix` method.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing data tensors of the input and output.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `max` or `fillMatrix` methods encounter an error, they may throw an exception depending on their implementation.
             *
             * @throws [Exception type from max or fillMatrix] If the `max` or `fillMatrix` methods encounter an error during execution.
             *
             * @note
             * - Ensure that the input and output tensors are properly initialized before calling this function.
             * - The time complexity of this function is O(b * c), where `b` is the batch size (`inputs[0]->output->shape()[0]`) and `c` is the number of channels (`inputs[0]->output->shape()[1]`).
             *
             * @code
             * ```cpp
             * GlobalMaxPoolNode globalMaxPoolNode(...); // Assume GlobalMaxPoolNode is properly initialized
             * globalMaxPoolNode.forward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void forward() override;

            /**
             * @brief Performs the backward pass of the global max - pooling operation.
             *
             * @param None
             *
             * @return None
             *
             * This function performs the backward pass of the global max - pooling operation. First, it checks if the output tensor of the input node requires gradient computation. If so, it retrieves the host data and gradients of the output tensor. Then, it iterates over each batch and channel of the input tensor's output. For each combination of batch index `i` and channel index `j`, it calculates an index `idx` and uses the `find` method to locate the position of the maximum value in the input tensor corresponding to the output value at `idx`. Finally, it sets the gradient at that position in the input tensor using the `setData` method.
             *
             * Memory management strategy: This function does not allocate or deallocate any memory directly. It operates on the existing data and gradient tensors of the input and output.
             * Exception handling mechanism: There is no explicit exception handling in this function. If the `hostData`, `hostGrad`, `find`, or `setData` methods encounter an error, they may throw an exception depending on their implementation.
             *
             * @throws [Exception type from hostData, hostGrad, find, or setData] If the `hostData`, `hostGrad`, `find`, or `setData` methods encounter an error during execution.
             *
             * @note
             * - Ensure that the input and output tensors and their gradients are properly initialized before calling this function.
             * - The time complexity of this function is O(b * c), where `b` is the batch size (`inputs[0]->output->shape()[0]`) and `c` is the number of channels (`inputs[0]->output->shape()[1]`).
             *
             * @code
             * ```cpp
             * GlobalMaxPoolNode globalMaxPoolNode(...); // Assume GlobalMaxPoolNode is properly initialized
             * globalMaxPoolNode.backward();
             * ```
             * @endcode
             *
             * @author
             * Mgepahmge(https://github.com/Mgepahmge)
             *
             * @date
             * 2024/07/15
             */
            void backward() override;
        };
    }

    /**
     * @namespace nz::nodes::loss
     * @brief Contains loss function nodes for computing various loss metrics in a machine learning model.
     *
     * The `Loss` namespace provides a collection of nodes that represent different loss functions used during the training
     * of machine learning models. These loss functions are used to evaluate the model's performance by calculating the
     * difference between the predicted output and the true values.
     *
     * This namespace includes:
     * - **Regression Loss Functions**: Nodes like `MeanSquaredErrorNode` that compute loss for regression tasks.
     * - **Classification Loss Functions**: Nodes such as `BinaryCrossEntropyNode` for computing loss in binary classification problems.
     *
     * Loss function nodes perform key roles in the optimization process by guiding the model to minimize errors during training.
     * These nodes integrate with the model’s computational graph, where they compute the forward pass loss and its gradients during the backward pass.
     *
     * @note
     * - The nodes in this namespace are specifically designed to handle loss calculations in supervised learning tasks.
     * - These loss functions are typically combined with optimization algorithms like Gradient Descent during model training.
     * - Ensure that the input tensors are compatible in terms of shape for proper loss computation.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    namespace loss {
        /**
 * @class MeanSquaredErrorNode
 * @brief Represents the Mean Squared Error (MSE) loss function node in a computational graph.
 *
 * The `MeanSquaredErrorNode` class computes the Mean Squared Error loss between two input tensors. The loss is
 * calculated as the average of the squared differences between the corresponding elements of the two tensors:
 * ```
 * MSE(x, y) = 1/n * Σ (x_i - y_i)^2
 * ```
 * where `x` and `y` are the two input tensors, and `n` is the number of elements in the tensors. This class is typically used
 * for training models, especially in regression tasks.
 *
 * @details
 * Key features:
 * - **Forward Pass**: Calculates the Mean Squared Error loss between the two input tensors, storing the result in the output tensor.
 *   The MSE loss is computed on a per-element basis and aggregated across the entire tensor.
 * - **Backward Pass**: Computes the gradients of the MSE loss with respect to the input tensors for use in backpropagation.
 *   The gradients are propagated only if the output tensor requires gradients.
 * - **Shape Compatibility**: Ensures that both input tensors have the same shape. An exception is thrown if the shapes do not match.
 * - **Efficient Computation**: The forward and backward passes are optimized for parallel execution using CUDA.
 *
 * This class is part of the `nz::nodes` namespace and is useful in models where Mean Squared Error is the loss function.
 *
 * @note
 * - The `MeanSquaredErrorNode` requires two input nodes, both of which must have tensors of the same shape.
 * - The forward pass performs the MSE calculation on the GPU, while the backward pass computes the gradient of the MSE loss.
 * - The loss is stored in the `loss` attribute, which is updated during the forward pass.
 * - The gradients are stored in the `grad` attribute of the input tensors during the backward pass.
 *
 * ### Usage Example:
 * ```cpp
 * // Example: Using MeanSquaredErrorNode in a computational graph
 * InputNode input1({3, 3}, true);  // Create the first input node with shape {3, 3}
 * InputNode input2({3, 3}, true);  // Create the second input node with shape {3, 3}
 *
 * float data1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};  // Sample input1 values
 * float data2[] = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f};  // Sample input2 values
 * input1.output->dataInject(data1);  // Copy data to the first input tensor
 * input2.output->dataInject(data2);  // Copy data to the second input tensor
 *
 * MeanSquaredErrorNode mse_node(&input1, &input2);  // Create the Mean Squared Error node
 * mse_node.forward();  // Perform the forward pass and compute the MSE loss
 * mse_node.backward();  // Perform the backward pass and compute gradients
 *
 * std::cout << "MSE Loss: " << mse_node.getLoss() << std::endl;  // Print the computed MSE loss
 * ```
 *
 * @see forward() for the Mean Squared Error computation in the forward pass.
 * @see backward() for gradient computation in the backward pass.
 *
 * @author
 * Mgepahmge (https://github.com/Mgepahmge)
 *
 * @date 2024/12/07
 */
        class DL_API MeanSquaredErrorNode : public io::OutputNode {
        public:
            /**
             * @brief Constructor to initialize a `MeanSquaredErrorNode` for computing the Mean Squared Error (MSE) loss.
             *
             * The constructor initializes a `MeanSquaredErrorNode`, which computes the MSE loss between two input nodes. It checks
             * that both input nodes have the same shape and sets up the necessary internal structures for the loss calculation.
             *
             * @param input1 A pointer to the first input node. The `output` tensor of this node will be used as the predicted values.
             * @param input2 A pointer to the second input node. The `output` tensor of this node will be used as the ground truth values.
             *
             * @throws std::invalid_argument if `input1` and `input2` do not have the same shape.
             *
             * @details
             * - The constructor verifies that the shapes of the two input tensors are the same. If they are not, an exception is thrown.
             * - The second input node (`input2`) is added to the `inputs` vector, while the first input node is inherited from the
             *   `OutputNode` base class.
             * - The `type` attribute is set to `"MeanSquaredError"`, indicating the type of operation this node represents.
             *
             * @note
             * - The MSE loss is only calculated if the input nodes have matching shapes. If the shapes do not match, an exception
             *   will be thrown to ensure consistency.
             * - The constructor initializes the internal state and prepares the node for the forward and backward passes.
             *
             * @see forward() for computing the MSE loss in the forward pass.
             * @see backward() for computing the gradients in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/07
             */
            explicit MeanSquaredErrorNode(Node* input1, Node* input2);

            /**
             * @brief Computes the forward pass of the Mean Squared Error (MSE) loss function.
             *
             * This method calculates the Mean Squared Error (MSE) loss between two input tensors. The loss is computed by
             * comparing the predicted values (`inputs[0]`) to the ground truth values (`inputs[1]`) element-wise. The result
             * is accumulated in the `loss` attribute of the node.
             *
             * The computation is performed in parallel on the GPU using CUDA kernels for efficiency.
             *
             * @details
             * - The method first calls the `forward()` method of the `OutputNode` base class to handle any initialization
             *   and setup required by the base class.
             * - CUDA is used to compute the MSE loss across the entire tensor. A kernel is launched to calculate the squared
             *   differences between the elements of the two input tensors.
             * - The results from the GPU computation are copied to the host memory, and the total MSE loss is accumulated.
             * - The final computed loss is stored in the `loss` attribute.
             *
             * @note
             * - The `inputs` vector must contain exactly two nodes: the predicted values and the ground truth values. Both
             *   tensors must have the same shape, and this is validated during the initialization.
             * - The loss is stored in the `loss` attribute, which is updated after each forward pass.
             *
             * @see backward() for the backward pass, which computes the gradients of the MSE loss.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/07
             */
            void forward() override;

            /**
             * @brief Computes the backward pass of the Mean Squared Error (MSE) loss function.
             *
             * This method computes the gradients of the MSE loss with respect to the input tensors (`inputs[0]` and `inputs[1]`).
             * The gradients are calculated only if the output tensor requires gradients, and they are used for backpropagation
             * in training deep learning models.
             *
             * The backward pass is performed using a CUDA kernel to efficiently compute the gradients in parallel on the GPU.
             *
             * @details
             * - The method first checks if the output tensor requires gradients by calling `requiresGrad()` on the output.
             * - If gradients are required, a CUDA kernel (`MSEBackward`) is launched to compute the gradients of the loss with
             *   respect to both input tensors.
             * - The gradients are computed by comparing the predicted values (`inputs[0]`) and the ground truth values (`inputs[1]`)
             *   in a parallel manner across all elements of the tensors.
             * - The result is stored in the `grad` attribute of the output tensor.
             *
             * @note
             * - The method assumes that both input tensors have the same shape, as validated during the initialization phase.
             * - The `requiresGrad()` check ensures that gradients are only computed if necessary, avoiding unnecessary computations.
             * - This method is essential for updating the model's parameters during the training process through backpropagation.
             *
             * @see forward() for the forward pass, which computes the MSE loss.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/07
             */
            void backward() override;
        };

        /**
         * @class BinaryCrossEntropyNode
         * @brief Represents the Binary Cross-Entropy (BCE) loss function node in a computational graph.
         *
         * The `BinaryCrossEntropyNode` class computes the Binary Cross-Entropy loss between two input tensors. BCE is typically
         * used in binary classification tasks to measure the difference between predicted probabilities and the true binary labels.
         * The loss is calculated as:
         * ```
         * BCE(x, y) = -1/n * Σ [y_i * log(x_i) + (1 - y_i) * log(1 - x_i)]
         * ```
         * where `x` represents the predicted probabilities (output of the model), and `y` represents the true binary labels
         * (either 0 or 1). The loss is computed element-wise for each pair of corresponding values in the tensors.
         *
         * @details
         * Key features:
         * - **Forward Pass**: Calculates the Binary Cross-Entropy loss between the two input tensors, storing the result in the output tensor.
         *   The BCE loss is computed for each element, and the results are aggregated to produce the final loss.
         * - **Backward Pass**: Computes the gradients of the BCE loss with respect to the input tensors for use in backpropagation.
         *   The gradients are propagated only if the output tensor requires gradients.
         * - **Shape Compatibility**: Ensures that both input tensors have the same shape. An exception is thrown if the shapes do not match.
         * - **Efficient Computation**: The forward and backward passes are optimized for parallel execution using CUDA to handle large datasets efficiently.
         *
         * This class is part of the `nz::nodes` namespace and is used in models for binary classification tasks.
         *
         * @note
         * - The `BinaryCrossEntropyNode` requires two input nodes: the predicted probabilities (`input1`) and the true binary labels (`input2`).
         * - Both input tensors must have the same shape, and the BCE loss is calculated element-wise across the tensors.
         * - The `forward()` method computes the BCE loss on the GPU, and the `backward()` method computes the gradients of the BCE loss.
         * - The loss is stored in the `loss` attribute, which is updated during the forward pass.
         * - The gradients are stored in the `grad` attribute of the output tensor during the backward pass.
         *
         * ### Usage Example:
         * ```cpp
         * // Example: Using BinaryCrossEntropyNode in a computational graph
         * InputNode input1({3, 3}, true);  // Create the first input node (predicted probabilities)
         * InputNode input2({3, 3}, true);  // Create the second input node (true labels)
         *
         * float data1[] = {0.9f, 0.2f, 0.8f, 0.1f, 0.5f, 0.7f, 0.3f, 0.9f, 0.6f};  // Sample predicted values
         * float data2[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f};  // Sample true labels
         * input1.output->dataInject(data1);  // Copy data to the first input tensor
         * input2.output->dataInject(data2);  // Copy data to the second input tensor
         *
         * BinaryCrossEntropyNode bce_node(&input1, &input2);  // Create the Binary Cross-Entropy node
         * bce_node.forward();  // Perform the forward pass and compute the BCE loss
         * bce_node.backward();  // Perform the backward pass and compute gradients
         *
         * std::cout << "BCE Loss: " << bce_node.getLoss() << std::endl;  // Print the computed BCE loss
         * ```
         *
         * @see forward() for the Binary Cross-Entropy computation in the forward pass.
         * @see backward() for gradient computation in the backward pass.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date 2024/12/07
         */
        class DL_API BinaryCrossEntropyNode : public io::OutputNode {
        public:
            /**
             * @brief Constructor to initialize a `BinaryCrossEntropyNode` for computing the Binary Cross-Entropy loss.
             *
             * The constructor initializes a `BinaryCrossEntropyNode`, which applies the Binary Cross-Entropy loss function
             * to two input tensors. It verifies that both input tensors have the same shape and establishes a connection in the
             * computational graph by storing the second input tensor. The node's type is set to "BinaryCrossEntropy".
             *
             * @param input1 A pointer to the first input node. This tensor represents the predicted probabilities.
             * @param input2 A pointer to the second input node. This tensor represents the true binary labels (0 or 1).
             *
             * @throws std::invalid_argument If the shapes of the two input tensors do not match.
             *
             * @details
             * - This constructor ensures that the input tensors have the same shape; otherwise, an exception is thrown.
             * - The first input tensor is connected to the output node, while the second input tensor is added to the `inputs` vector.
             * - The node's type is set to "BinaryCrossEntropy" to reflect its operation.
             *
             * @note
             * - The two input tensors must have the same shape, as the Binary Cross-Entropy loss is computed element-wise.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/07
             */
            explicit BinaryCrossEntropyNode(Node* input1, Node* input2);

            /**
             * @brief Computes the Binary Cross-Entropy (BCE) loss in the forward pass.
             *
             * This method computes the Binary Cross-Entropy loss between the predicted probabilities (from the first input tensor)
             * and the true binary labels (from the second input tensor). The loss is calculated element-wise and accumulated.
             * The result is stored in the `loss` attribute, which can be accessed after the forward pass.
             *
             * The Binary Cross-Entropy loss is computed as:
             * ```
             * BCE(y_pred, y_true) = - ( y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred) )
             * ```
             * where `y_pred` is the predicted probability and `y_true` is the true label (0 or 1).
             *
             * The calculation is done in parallel on the GPU using CUDA to handle large tensor sizes efficiently.
             *
             * @details
             * - The forward pass involves allocating memory on the GPU, performing the BCE loss computation, and accumulating the loss.
             * - A kernel is launched with a grid of threads to compute the loss across all elements of the tensors.
             * - The result is copied back to the host memory, where the accumulated loss is added to the `loss` attribute.
             * - After computation, the allocated GPU memory is freed.
             *
             * @note
             * - This method assumes that both input tensors have the same shape. The computation is performed element-wise, and
             *   the tensors must be compatible for this operation.
             * - The `loss` attribute will hold the accumulated Binary Cross-Entropy loss after the forward pass.
             *
             * @see backward() for the gradient computation in the backward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/07
             */
            void forward() override;

            /**
             * @brief Computes the gradients of the Binary Cross-Entropy (BCE) loss with respect to the inputs during the backward pass.
             *
             * This method computes the gradients of the Binary Cross-Entropy loss with respect to both input tensors (`input1` and `input2`).
             * The gradients are computed only if the output tensor requires gradients (i.e., during the backpropagation process).
             * The gradients are propagated back to the input nodes to update their weights during training.
             *
             * The gradient of Binary Cross-Entropy with respect to the predicted probabilities (`y_pred`) is computed as:
             * ```
             * dBCE/dy_pred = - ( y_true / y_pred ) + ( (1 - y_true) / (1 - y_pred) )
             * ```
             * where `y_pred` is the predicted probability and `y_true` is the true binary label (0 or 1).
             *
             * The gradient computation is parallelized on the GPU using CUDA, enabling efficient backpropagation even with large datasets.
             *
             * @details
             * - A kernel is launched to compute the gradients in parallel for all elements in the tensors.
             * - The gradients are stored in the `grad` attribute of the output tensor, which is propagated to the input nodes during backpropagation.
             * - The backward pass is only executed if the output tensor has `requiresGrad()` set to true, ensuring that gradients are computed only when necessary.
             * - The method uses GPU memory for efficient computation and returns the results to the host after calculation.
             *
             * @note
             * - This method assumes that both input tensors have the same shape. The gradients are computed element-wise and
             *   are propagated back to the input tensors accordingly.
             * - The gradients with respect to the predicted probabilities (`y_pred`) are accumulated in the `grad` attribute of the output tensor.
             * - If the output does not require gradients, this method does nothing.
             *
             * @see forward() for the BCE loss computation in the forward pass.
             *
             * @author
             * Mgepahmge (https://github.com/Mgepahmge)
             *
             * @date 2024/12/07
             */
            void backward() override;
        };
    }
}

#endif //NODES_CUH

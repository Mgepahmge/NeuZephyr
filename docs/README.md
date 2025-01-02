/** @mainpage

# Welcome to NeuZephyr! 🎉

NeuZephyr is a lightweight deep learning library developed in C++ with CUDA C, designed to provide efficient GPU acceleration for deep learning model training and inference. Its goal is to help developers quickly implement deep learning models while maintaining an easy-to-use interface.

If you're new to NeuZephyr and want to get started quickly, **the best place to begin is the documentation for the `ComputeGraph` class**. The `ComputeGraph` class allows you to define and manipulate the structure of your neural network, making it the foundation for both training and inference.

Once you understand how to use the **ComputeGraph**, you'll be able to easily integrate other features like Tensors, Nodes, Optimizers, and more into your models.

For a detailed guide on how to use the `ComputeGraph` class, please refer to the **`ComputeGraph.cuh`** file or its documentation.

## Features

- Built on CUDA C for efficient GPU acceleration
- Supports common deep learning operations, such as tensor operations, matrix multiplication, etc.
- Lightweight design, easy to integrate into existing projects
- C++ interface for seamless integration with other C++ projects

## Installation

### System Requirements

- CUDA driver (Recommended version: CUDA 10.2 or higher)
- CMake 3.10 or higher

### Installation Steps

1. Clone the project:
   ```bash
   git clone https://github.com/Mgepahmge/NeuZephyr.git
   cd NeuZephyr
   ```

2. Create a build directory and compile the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   sudo make install
   ```

3. Use `NeuZephyr` in your CMake project:
   ```cmake
   find_package(NeuZephyr REQUIRED)
   target_link_libraries(your_project PRIVATE NeuZephyr::NeuZephyr)
   ```

### Dependencies

- CUDA driver
- CMake 3.10 or higher

## Key Components

### Tensor: The Core Data Structure

A **Tensor** is the fundamental data structure in NeuZephyr. It represents multi-dimensional arrays of data, which are essential for storing and manipulating data in deep learning models. Tensors are used in all computations, including matrix operations, activation functions, and more.

Tensors in NeuZephyr support a wide range of operations such as addition, multiplication, and broadcasting. The `nz::data::Tensor` class is designed to be efficient and is optimized for use with CUDA, ensuring that tensor operations are performed on the GPU for faster computation.

For more detailed documentation on how to use the `Tensor` class, see the `Tensor.cuh` file.

### Node: The Compute Graph Node

A **Node** in NeuZephyr represents a unit of computation in the computational graph. It can be an input/output (I/O) node, an operation node (such as matrix multiplication), or a loss function node (e.g., Mean Squared Error or Cross-Entropy). Nodes connect together to form a computational graph, which is used to define the flow of data and operations in a neural network.

Nodes are defined within the `nz::nodes` namespace. Each type of node has specific functionality, and they work together to perform the computations required for training and inference.

For more detailed documentation on the different node types, refer to the `Nodes.cuh` file or the `nz::nodes` namespace documentation.

### Optimizer: The Optimization Algorithm

An **Optimizer** is responsible for adjusting the parameters of the model (such as weights and biases) during training to minimize the loss function. NeuZephyr includes several commonly used optimizers, such as **Stochastic Gradient Descent (SGD)** and **Adam**.

Optimizers play a crucial role in the training process, and different optimizers can be more suitable for different tasks. The `nz::opt` namespace contains classes for implementing various optimization algorithms.

For detailed documentation on each optimizer and how to configure them, please refer to the `Optimizer.cuh` file or the `nz::opt` namespace documentation.

### ComputeGraph: The Computational Graph

The **ComputeGraph** is the backbone of any deep learning model in NeuZephyr. It defines the structure of the neural network, including the layers, operations, and data flow. Each node in the graph represents a computation, and the connections between them define the execution order.

If you are looking to build your own neural network using NeuZephyr, you will need to understand how to create and manipulate a `ComputeGraph`. This class allows you to define complex models and perform training or inference on them. It also handles automatic differentiation for backpropagation during training.

For a deeper dive into how to use the `ComputeGraph` class and build custom neural networks, refer to the documentation for the `ComputeGraph.cuh` file.

---

For more detailed information, please refer to the corresponding header files or the documentation within each class/namespace. If you are new to NeuZephyr, we recommend starting with the `Tensor` and `ComputeGraph` documentation to get an understanding of how the core components work together.

For further assistance, feel free to check the repository or reach out to the maintainers.

### Notes

- The library currently supports only CUDA environments and assumes that the CUDA driver and libraries are properly installed.
- Ensure that your GPU environment is compatible with the required CUDA version.

---

### Contact & Additional Information

For inquiries, issues, or contributions, please contact:

- Email: [yss489589139@outlook.com](mailto:yss489589139@outlook.com)
- GitHub Repository: [https://github.com/Mgepahmge/NeuZephyr](https://github.com/Mgepahmge/NeuZephyr)

### Disclaimer

This project is provided strictly for learning and research purposes. It is not intended for production use or commercial deployment.

---

### License

This project is licensed under the **MIT License**.

See the [LICENSE](https://github.com/Mgepahmge/NeuZephyr/blob/main/LICENSE) file for details.

---

## Changelog

### v0.1.0 - Initial Release
- First release of NeuZephyr.
- Supported basic matrix operations (tensor multiplication, addition, etc.).
- Implemented most common activation functions (ReLU, Sigmoid, Tanh, etc.).
- Added basic optimizers (SGD, Adam).
- Implemented a linear layer for neural networks.
- Supported common loss functions (Mean Squared Error, Cross-Entropy, etc.).
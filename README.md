# NeuZephyr

NeuZephyr is a lightweight deep learning library developed in C++ with CUDA C, designed to provide efficient GPU acceleration for deep learning model training and inference. Its goal is to help developers quickly implement deep learning models while maintaining an easy-to-use interface.

---

## Features

- Built on CUDA C for efficient GPU acceleration
- Supports common deep learning operations, such as tensor operations, matrix multiplication, etc.
- Lightweight design, easy to integrate into existing projects
- C++ interface for seamless integration with other C++ projects

---

## Installation

### System Requirements

- CUDA driver (Recommended version: CUDA 12 or higher)
- CMake 3.10 or higher (Recommended version: CMake 3.18 or higher)

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

- **CUDA Driver**: A GPU with compute capability 7.0 or higher is required to support Tensor Cores. Tensor Cores are available on NVIDIA Volta, Turing, and Ampere architectures (e.g., V100, T4, A100, and others).
- **CUDA Version**: CUDA 10.1 or higher is required to access Tensor Cores for single-precision matrix multiplication. Ensure the appropriate driver and runtime libraries are installed for full functionality.
- **CMake**: Version 3.18 or higher is required for building the project.

---

## Example

### C++ Example Code using `NeuZephyr`

```cpp
#include <iostream>
#include <NeuZephyr/ComputeGraph.cuh>

int main() {
   // Create a compute graph
   graph::ComputeGraph graph;
   
   // Add input nodes
   graph.addInput({3, 4}, false, "Input");
   graph.addInput({4, 3}, true, "Weight");
   graph.addInput({3, 3}, false, "Label");
   
   // Add other nodes
   graph.addNode("MatMul", "Input", "Weight", "MatMul");
   graph.addNode("ReLU", "MatMul", "", "ReLU");
   graph.addNode("MeanSquaredError", "ReLU", "Label");
   
   graph.randomizeAll();
   
   // Perform forward and backward passes
   graph.forward();
   graph.backward();
   std::cout << graph << std::endl; // Print result
   
   // Update weights
   opt::SGD optimizer(0.01); // Create optimizer
   graph.update(&optimizer); // Update weights
}
```

### Notes

- The library currently supports only CUDA environments and assumes that the CUDA driver and libraries are properly installed.
- Ensure that your GPU environment is compatible with the required CUDA version.

---

## Documentation

For detailed documentation on how to use NeuZephyr, please visit the [NeuZephyr Documentation](https://mgepahmge.github.io/NeuZephyrDoc/).

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

### v0.2.0 - Performance Optimization
- Optimized performance by applying thread bundle shuffling to accumulative kernel functions, resulting in a 13% performance boost on the author's device.
- Integrated Tensor Cores for half-precision fast matrix multiplication, leading to a 20% performance improvement on the author's device.
- Further fine-tuned matrix operations for increased efficiency.


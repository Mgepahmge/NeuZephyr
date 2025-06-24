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

2. Build and install the project using CMake (Release mode):
   ```bash
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --config Release
   sudo cmake --install .  # Omit 'sudo' on Windows
   ```
   **Add to PATH** (Windows):
   After installation, add `<INSTALL_DIR>/NeuZephyr/bin` to your system's `PATH` environment variable.


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
class SegmentationModel : public Model {
public:
    InputNode input{{10,3,1024,1024}};  // Batch initialized directly
    InputNode target;
 
    SegmentationModel() : target({10,1,8,1}) {
        auto x = Conv2d(&input, 1, 3, 3, 1, 1);
        x = ReLU(x);
        x = Conv2d(x, 1, 3, 3, 1, 1);
        x = AvgPool2d(x, 5, 2);
        x = Linear(x, 16);
        x = Softmax(x);
        BCELoss(x, &target);  // Graph termination
    }
};
 
int main() {
    SegmentationModel model;
    model.input = load_tensor(...);
    model.target = load_labels(...);
 
    opt::Adam optimizer(0.01, 0.9, 0.999);
    for(int epoch = 0; epoch < 100; ++epoch) {
        model.forward();
        model.backward();
        model.update(&optimizer);
        std::cout << "Loss: " << model.getLoss() << std::endl;
    }
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

### v0.6 - Model Class & Neural Automation Framework
- Introduced **`Model` base class** enabling declarative model construction via inheritance, with automatic parameter tracking and gradient graph generation.
- Added high-level neural components (`Conv2D`, `MaxPool`, `Linear`) with built-in auto-differentiation, eliminating manual computation graph assembly.

### v0.5 - High-Dimensional Tensor Infrastructure Overhaul
- Upgraded core tensor infrastructure from 2D to **fourth-order tensor** representation, enabling native support for high-dimensional data structures.
- Refactored core tensor modules (e.g., memory allocation, shape propagation) to align with the new four-dimensional paradigm while maintaining backward compatibility.
- Optimized memory layout and access patterns for fourth-order tensors, improving cache utilization and computational efficiency.
- Preserved low-dimensional tensor interfaces (2D/3D) via automatic view projection for scenarios requiring legacy dimensionality.

### v0.4 - CUDA Stream Management Enhancement
- Added CUDA Stream Manager to automatically distribute CUDA operations across multiple streams, enabling asynchronous execution and improved concurrency.
- Unified stream-aware APIs for seamless integration with existing Tensor and MappedTensor classes, ensuring backward compatibility.
- Optimized dependency resolution to intelligently track and synchronize inter-stream operations, minimizing synchronization overhead.
- Preserved explicit stream control for advanced users requiring fine-grained parallelism management (e.g., custom kernel launches).

### v0.3 - Memory Management Upgrade
- Added MappedTensor class using CUDA zero-copy memory (ideal for frequent host-side data access in non-compute-intensive scenarios).
- Unified APIs between MappedTensor and original Tensor class (using CUDA global memory) for consistent interfaces.
- Preserved high-performance Tensor implementation (recommended for compute-intensive workloads).
- Optimized memory mapping mechanisms to reduce host-device transfer overhead.

### v0.2 - Performance Optimization
- Optimized performance by applying thread bundle shuffling to accumulative kernel functions, resulting in a 13% performance boost on the author's device.
- Integrated Tensor Cores for half-precision fast matrix multiplication, leading to a 20% performance improvement on the author's device.
- Further fine-tuned matrix operations for increased efficiency.

### v0.1 - Initial Release
- First release of NeuZephyr.
- Supported basic matrix operations (tensor multiplication, addition, etc.).
- Implemented most common activation functions (ReLU, Sigmoid, Tanh, etc.).
- Added basic optimizers (SGD, Adam).
- Implemented a linear layer for neural networks.
- Supported common loss functions (Mean Squared Error, Cross-Entropy, etc.).


# NeuZephyr

NeuZephyr is a lightweight deep learning library developed in C++ with CUDA C, designed to provide efficient GPU acceleration for deep learning model training and inference. Its goal is to help developers quickly implement deep learning models while maintaining an easy-to-use interface.

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

## Example

### C++ Example Code using `NeuZephyr`

```cpp
#include <iostream>
#include <NeuZephyr/ComputeGraph.cuh>

int main() {
    nz::data::Tensor a({3, 4});
    nz::data::Tensor b({4, 3});
    a.fill(1);
    b.fill(2);
    std::cout << a * b << std::endl;
    return 0;
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
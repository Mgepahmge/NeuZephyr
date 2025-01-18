/**
 * @file Tensor.cuh
 * @brief Definition of the Tensor class for GPU-based tensor operations.
 *
 * This file provides the declaration of the `Tensor` class, which is a key data structure for
 * representing multidimensional arrays (tensors) in GPU memory. It supports mathematical
 * operations, memory management, and utility functions for deep learning applications. The
 * implementation leverages CUDA for high-performance computations.
 *
 * @details
 * The `Tensor` class encapsulates the following key features:
 * - **Memory Management**: Efficient allocation and deallocation of GPU memory using `cudaMalloc`
 *   and `cudaMemcpy`.
 * - **Mathematical Operations**: Overloaded operators and utility functions for common tensor
 *   operations, including element-wise arithmetic, reshaping, and transposing.
 * - **Gradient Support**: Tracks gradients for tensors that require gradient computation, supporting
 *   backpropagation in neural networks.
 * - **Flexible Initialization**: Provides constructors for various initialization methods, including
 *   shape-based, iterator-based, and initializer-list-based creation.
 * - **Activation Functions**: Includes common activation functions such as ReLU, Sigmoid, Tanh,
 *   and advanced functions like Swish and HardSwish.
 *
 * This class is part of the `nz::data` namespace and is designed for extensibility and
 * high performance in machine learning workflows.
 *
 * @note
 * Ensure proper error handling and memory cleanup when using this class to avoid GPU memory
 * leaks or runtime errors.
 *
 * @author
 * Mgepahmge(https://github.com/Mgepahmge)
 *
 * @date
 * 2024/11/29
 */
#ifndef TENSOR_CUH
#define TENSOR_CUH

#include "OperationKernels.cuh"
#include <iterator>
#include <stdexcept>
#include <vector>
#include "dl_export.cuh"

/**
 * @namespace nz::data
 * @brief Contains data structures and utilities for tensor operations in machine learning workflows.
 *
 * The `nz::data` namespace provides foundational classes and functions
 * for managing and manipulating tensors in GPU-based computations. It is designed
 * for use in deep learning frameworks and other numerical computing applications.
 *
 * @details
 * Key components within this namespace include:
 * - **Tensor**: A class representing multidimensional arrays (tensors) stored in GPU memory.
 * - **Utilities**: Functions and operators for performing mathematical operations,
 *   memory management, and activation functions.
 *
 * The namespace is intended to encapsulate all tensor-related functionality to ensure
 * modularity and maintainability in the larger nz project.
 *
 * @note
 * The components in this namespace rely on CUDA for GPU-based operations. Ensure that
 * CUDA-compatible hardware and software are properly configured.
 *
 * @author
 * Mgepahmge(https://github.com/Mgepahmge)
 *
 * @date
 * 2024/11/29
 */
namespace nz::data {
    /**
     * @class Tensor
     * @brief A class for representing and manipulating multidimensional arrays (tensors) in GPU memory.
     *
     * The `Tensor` class is designed for high-performance numerical computations in GPU-based
     * environments. It provides a wide range of functionalities, including tensor creation,
     * mathematical operations, memory management, and gradient computation for deep learning tasks.
     *
     * ### Type Definitions:
     * - `size_type`: An alias for `unsigned long long`, used to represent the size of the tensor.
     *   Supports large tensors with up to 64-bit indices.
     * - `value_type`: An alias for `float`, representing the data type of the tensor elements.
     *   Suitable for most machine learning computations.
     * - `shape_type`: An alias for `std::vector<int>`, representing the shape of the tensor
     *   (e.g., `{2, 3}` for a 2x3 matrix).
     *
     * @details
     * ### Key Features:
     * - **Memory Management**: Handles GPU memory allocation and deallocation using CUDA.
     * - **Flexible Initialization**: Supports initialization via shapes, data pointers,
     *   initializer lists, and iterators.
     * - **Mathematical Operations**: Includes overloaded operators (`+`, `-`, `*`, `/`)
     *   and activation functions (`ReLU`, `Sigmoid`, `Tanh`, etc.).
     * - **Gradient Support**: Tracks gradients for tensors that require gradient computation
     *   (`requires_grad`) to facilitate backpropagation in neural networks.
     * - **Shape Transformation**: Supports reshaping and transposing tensors.
     *
     * ### Usage Example:
     * ```cpp
     * using namespace nz::data;
     *
     * // Create a tensor that requires gradient with shape 2x3
     * Tensor tensor({2, 3}, true);
     * tensor.fill(1.0f);     // Fill the tensor with value 1.0
     *
     * // Apply element-wise ReLU activation
     * Tensor result = ReLU(tensor);
     * std::cout << "ReLU activated tensor:" << std::endl;
     * std::cout << result << std::endl;        // Print the result of ReLU activation
     *
     * // Perform matrix multiplication (2x3 * 3x2 = 2x2)
     * Tensor tensor3({3, 2}, true);
     * tensor3.fill(3.0f); // Fill tensor3 with value 3.0
     * Tensor multiplied_result = tensor * tensor3;  // Multiply tensor (2x3) by tensor3 (3x2)
     * std::cout << "Multiplication result (2x3 * 3x2 = 2x2):" << std::endl;
     * std::cout << multiplied_result << std::endl;  // Print the result of matrix multiplication
     * ```
     *
     *
     * @note
     * - Ensure proper cleanup by calling the destructor or relying on RAII to avoid memory leaks.
     * - Tensor size and shape must match during operations to prevent runtime errors.
     * - Requires CUDA-compatible hardware and a properly configured environment.
     *
     * @author
     * Mgepahmge(https://github.com/Mgepahmge)
     *
     * @date
     * 2024/11/29
     */
    class DL_API Tensor {
    public:
        using size_type = unsigned long long;
        using value_type = float;
        using shape_type = std::vector<int>;

        friend DL_API std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        friend DL_API std::istream& operator>>(std::istream& is, const Tensor& tensor);
        friend DL_API Tensor operator*(value_type lhs, const Tensor& rhs);
        friend DL_API Tensor operator*(const Tensor& lhs, value_type rhs);
        friend DL_API Tensor operator/(const Tensor& lhs, value_type rhs);
        friend DL_API Tensor operator+(const Tensor& lhs, value_type rhs);
        friend DL_API Tensor operator+(value_type lhs, const Tensor& rhs);
        friend DL_API Tensor operator-(const Tensor& lhs, value_type rhs);
        friend DL_API Tensor operator-(value_type lhs, const Tensor& rhs);
        friend DL_API Tensor ReLU(const Tensor& tensor);
        friend DL_API Tensor Sigmoid(const Tensor& tensor);
        friend DL_API Tensor Tanh(const Tensor& tensor);
        friend DL_API Tensor LeakyReLU(const Tensor& tensor, float alpha = 0.01f);
        friend DL_API Tensor Swish(const Tensor& tensor);
        friend DL_API Tensor ELU(const Tensor& tensor, float alpha = 1.0f);
        friend DL_API Tensor HardSigmoid(const Tensor& tensor, float alpha = 0.2f, float beta = 0.5f);
        friend DL_API Tensor HardSwish(const Tensor& tensor, float alpha = 0.2f, float beta = 0.5f);
        friend DL_API Tensor Softmax(const Tensor& tensor);

        /// @name Constructors and Destructors
        /// @{

        /**
        * @brief Default constructor for Tensor.
        *
        * Initializes an empty Tensor with no data or shape. This constructor is primarily
        * used as a placeholder or for initializing variables before assigning a valid tensor.
        */
        Tensor();

        /**
        * @brief Constructor that initializes a Tensor with the specified shape.
        *
        * @param shape A vector representing the dimensions of the tensor.
        * @param requires_grad A boolean indicating whether the tensor requires gradient computation.
        *
        * This constructor allocates GPU memory for the tensor based on the specified shape.
        * If `requires_grad` is set to true, additional memory is allocated for storing gradients.
        */
        explicit Tensor(const shape_type& shape, const bool requires_grad = false);

        /**
        * @brief Constructor that initializes a Tensor with the specified shape and data.
        *
        * @param shape A vector representing the dimensions of the tensor.
        * @param data A pointer to the initial data to be copied into the tensor.
        * @param requires_grad A boolean indicating whether the tensor requires gradient computation.
        *
        * This constructor allocates GPU memory for the tensor and copies the provided data
        * to the GPU. If `requires_grad` is true, additional memory is allocated for gradients.
        */
        explicit Tensor(const shape_type& shape, const value_type* data, const bool requires_grad = false);

        /**
        * @brief Constructor that initializes a Tensor using an initializer list for shape.
        *
        * @param shape An initializer list representing the dimensions of the tensor.
        * @param requires_grad A boolean indicating whether the tensor requires gradient computation.
        *
        * This constructor is a convenience function for easily creating tensors without
        * explicitly constructing a `std::vector` for the shape.
        */
        explicit Tensor(const std::initializer_list<int>& shape, const bool requires_grad = false);

        /**
        * @brief Constructor that initializes a Tensor using an initializer list for shape.
        *
        * @param shape An initializer list representing the dimensions of the tensor.
        * @param requires_grad A boolean indicating whether the tensor requires gradient computation.
        *
        * This constructor is a convenience function for easily creating tensors without
        * explicitly constructing a `std::vector` for the shape and copying the provided data.
        *
        */
        explicit Tensor(const std::initializer_list<int>& shape, const value_type* data,
                        const bool requires_grad = false);

        /**
        * @brief Template constructor to initialize a Tensor with data from an iterator range.
        *
        * @tparam Iterator The type of iterator used to provide data.
        * @param shape A vector representing the dimensions of the tensor.
        * @param first An iterator pointing to the beginning of the data.
        * @param last An iterator pointing to the end of the data.
        * @param requires_grad A boolean indicating whether the tensor requires gradient computation.
        *
        * This constructor calculates the total size of the tensor from the iterator range and
        * validates it against the provided shape. It then allocates memory on the GPU and
        * copies the data.
        *
        * @throws std::invalid_argument If the shape and data size do not match.
        */
        template <typename Iterator>
        Tensor::Tensor(const shape_type shape, Iterator first, Iterator last, const bool requires_grad) :
            _size(std::distance(first, last)), _shape(shape), _requires_grad(requires_grad) {
            if (shape[0] * shape[1] != _size) {
                throw std::invalid_argument("The size of the data does not match the shape.");
            }
            cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
            cudaMemcpy(_data, first, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
            if (_requires_grad) {
                cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
            }
        }

        /**
        * @brief Template constructor to initialize a Tensor using an initializer list and an iterator range.
        *
        * @tparam Iterator The type of iterator used to provide data.
        * @param shape An initializer list representing the dimensions of the tensor.
        * @param first An iterator pointing to the beginning of the data range.
        * @param last An iterator pointing to the end of the data range.
        * @param requires_grad A boolean indicating whether the tensor requires gradient computation.
        *
        * This constructor computes the size of the tensor from the iterator range and validates
        * it against the provided shape. Memory is allocated on the GPU for the data and gradients
        * (if applicable). The data is copied from the iterator range to the GPU.
        *
        * @throws std::invalid_argument If the shape dimensions do not match the size of the data range.
        *
        * @note The shape must be valid for the provided data size (i.e., the total elements in the
        * shape must equal the distance between `first` and `last`). CUDA memory (`cudaMalloc`) is
        * allocated, so ensure that GPU resources are available.
        */
        template <typename Iterator>
        Tensor::Tensor(const std::initializer_list<int>& shape, Iterator first, Iterator last,
                       const bool requires_grad) :
            _size(std::distance(first, last)), _shape(shape), _requires_grad(requires_grad) {
            if (_shape[0] * _shape[1] != _size) {
                throw std::invalid_argument("The size of the data does not match the shape.");
            }
            cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
            cudaMemcpy(_data, first, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
            if (_requires_grad) {
                cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
            }
        }

        /**
        * @brief Copy constructor for Tensor.
        *
        * @param other The Tensor object to copy from.
        *
        * Performs a deep copy of the tensor, including its shape, data, and gradient (if applicable).
        */
        Tensor(const Tensor& other);

        /**
        * @brief Move constructor for Tensor.
        *
        * @param other The Tensor object to move from.
        *
        * Moves the tensor data and ownership of the GPU memory to the new object.
        */
        Tensor(Tensor&& other) noexcept;

        /**
        * @brief Assignment operator for Tensor.
        *
        * @param other The Tensor object to assign from.
        *
        * Performs a deep copy of the tensor, including its shape, data, and gradient (if applicable).
        *
        * @return A reference to the assigned Tensor object.
        */
        Tensor& operator=(const Tensor& other);

        /**
        * @brief Move assignment operator for Tensor.
        *
        * @param other The Tensor object to move from.
        *
        * Moves the tensor data and ownership of the GPU memory to the new object.
        *
        * @return A reference to the assigned Tensor object.
        */
        Tensor& operator=(Tensor&& other) noexcept;

        /**
        * @brief Destructor for Tensor.
        *
        * Releases all GPU memory allocated for the tensor's data and gradient. Ensures
        * that no memory leaks occur during the lifetime of the Tensor object.
        */
        ~Tensor();

        /// @}

        /// @name Getters and Setters
        /// @{

        /**
        * @brief Checks whether the tensor requires gradient computation.
        *
        * @return `true` if the tensor requires gradient computation, `false` otherwise.
        *
        * This function allows you to query whether the tensor is marked for gradient tracking,
        * which is essential for backpropagation in neural networks. By default, tensors do not
        * require gradients unless explicitly specified during construction or via `setRequiresGrad`.
        */
        bool requiresGrad() const noexcept;

        /**
        * @brief Retrieves the shape of the tensor.
        *
        * @return A `shape_type` (alias for `std::vector<int>`) representing the dimensions of the tensor.
        *
        * The shape provides information about the size of each dimension in the tensor. For example,
        * a tensor with shape `{2, 3}` represents a 2x3 matrix. The shape is defined during construction
        * or reshaping of the tensor.
        */
        shape_type shape() const noexcept;

        /**
        * @brief Retrieves the total number of elements in the tensor.
        *
        * @return A `size_type` (alias for `unsigned long long`) representing the total number of elements.
        *
        * This function calculates the product of the dimensions in the tensor's shape. For example,
        * a tensor with shape `{2, 3}` will have a size of 6. This value is useful for memory allocation
        * and tensor operations.
        */
        size_type size() const noexcept;

        /**
        * @brief Sets whether the tensor requires gradient computation.
        *
        * @param requires_grad A boolean indicating whether gradient computation is required.
        *
        * This function allows you to enable or disable gradient tracking for the tensor. If
        * gradient computation is enabled, additional memory may be allocated for storing gradients.
        *
        * @note Modifying this setting does not affect any existing gradient data stored in the tensor.
        */
        void setRequiresGrad(const bool requires_grad) noexcept;

        /**
         * @brief Retrieves a pointer to the tensor's data stored in GPU memory.
         *
         * @return A `value_type*` (pointer to float) pointing to the tensor's data in GPU memory.
         *
         * This function provides direct access to the raw data of the tensor stored in GPU memory.
         * It is useful for low-level operations or when interfacing with other libraries that
         * require access to the tensor's memory.
         *
         * @note
         * - The returned pointer points to GPU memory, so it cannot be directly dereferenced in CPU code.
         * - Ensure that CUDA synchronization is handled properly before using this pointer in GPU operations.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});
         * const float* gpu_data = tensor.data(); // Access raw data
         * // Use gpu_data in CUDA kernels or other GPU-based operations
         * ```
         * @endcode
         */
        value_type* data() const noexcept;

        /**
        * @brief Retrieves a pointer to the gradient data stored in GPU memory.
        *
        * @return A `value_type*` (pointer to float) pointing to the tensor's gradient data in GPU memory.
        *
        * This function provides access to the gradient data of the tensor, stored in GPU memory.
        * If the tensor does not require gradient computation (`requires_grad` is `false`),
        * the function throws a `std::runtime_error`.
        *
        * @throws std::runtime_error If the tensor does not require gradient computation.
        *
        * @note
        * - The returned pointer points to GPU memory and cannot be directly dereferenced in CPU code.
        * - Ensure that CUDA synchronization is handled properly before using this pointer in GPU operations.
        *
        * @code
        * ```cpp
        * Tensor tensor({2, 3}, true); // Create a tensor that requires gradients
        * try {
        *     const float* grad_data = tensor.grad(); // Access raw gradient data
        *     // Use grad_data in CUDA kernels or other GPU-based operations
        * } catch (const std::runtime_error& e) {
        *     std::cerr << e.what() << std::endl; // Handle error if tensor does not require gradients
        * }
        * ```
        * @endcode
        */
        value_type* grad() const;

        /**
         * @brief Copies data into the tensor from a raw pointer and reallocates memory.
         *
         * @param data A pointer to the raw data to be copied into the tensor. The data should be stored in
         *             memory accessible by the host (CPU).
         *
         * This function first frees any previously allocated memory for both the tensor's data and gradients
         * (if applicable). Then, it allocates new memory on the GPU for the tensor's data and, if the tensor
         * requires gradients, for its gradient data. The provided data is copied into the tensor's GPU memory.
         *
         * @note
         * - This function assumes that the provided data is stored in host memory (CPU). If the data is already
         *   in GPU memory, a different method should be used.
         * - It performs both memory allocation and data copy, ensuring that the tensor and its gradient memory
         *   are correctly managed.
         * - If gradients are required (`requires_grad`), the gradient memory is initialized to zero after allocation.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a 2x3 tensor
         * float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
         * tensor.copyData(input_data);  // Copy the data into the tensor
         * ```
         * @endcode
         */
        void copyData(const value_type* data);

        /**
         * @brief Copies gradient data from host to GPU memory.
         *
         * This function copies the provided gradient data from host (CPU) memory to the
         * tensor's gradient memory on the GPU. It assumes that the tensor has been initialized
         * with gradient support (`requires_grad` is `true`). If gradients are not required,
         * an exception will be thrown.
         *
         * @param grad A pointer to the gradient data in host (CPU) memory to be copied to the tensor's GPU memory.
         *
         * This function performs the following steps:
         * 1. It checks whether the tensor requires gradients. If not, a `std::runtime_error` is thrown.
         * 2. If gradients are required, it copies the provided gradient data from host memory to device memory using `cudaMemcpy`.
         *
         * @throws std::runtime_error If the tensor does not require gradients.
         *
         * @note
         * - Ensure that the `grad` pointer points to valid memory on the host (CPU).
         * - The tensor must have been created with `requires_grad` set to `true`. Otherwise, calling this function
         *   will result in an error.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3}, true);  // Create a tensor with gradient support
         * float grad_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
         * tensor.copyGrad(grad_data);  // Copy gradient data to tensor's GPU memory
         * ```
         * @endcode 
         */
        void copyGrad(const value_type* grad) const;

        /// @}

        /// @name Modifiers
        /// @{

        /**
         * @brief Resets the gradient data to zero.
         *
         * This function sets the gradient data of the tensor to zero. It is typically used
         * during training in neural networks to clear the gradients before the next backpropagation
         * pass. The gradient memory will remain allocated, but its contents will be zeroed out.
         *
         * @note
         * - This function does not deallocate the gradient memory; it only resets the stored gradient values.
         * - The tensor must have been created with `requires_grad` set to `true`, otherwise the function does nothing.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3}, true);  // Create a tensor with gradient support
         * tensor.zeroGrad();  // Reset the gradients to zero
         * ```
         * @endcode
         */
        void zeroGrad() const noexcept;

        /**
         * @brief Randomizes the tensor's data with a uniform distribution.
         *
         * This function fills the tensor's data with random values sampled from a uniform distribution
         * in the range [0, 1). The random number generator is initialized using the specified seed to ensure
         * reproducibility. The function uses the `curand` library to generate random numbers on the GPU.
         *
         * @param seed A `unsigned long long` value used to initialize the random number generator.
         *             The same seed will produce the same sequence of random numbers, ensuring reproducibility.
         *
         * This function performs the following steps:
         * 1. It creates a random number generator using `curandCreateGenerator`.
         * 2. It sets the seed for the random number generator using `curandSetPseudoRandomGeneratorSeed`.
         * 3. It generates uniform random numbers in the range [0, 1) and fills the tensor's data with these values.
         *
         * @note
         * - The generated random numbers are uniformly distributed in the range [0, 1).
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a tensor with shape 2x3
         * tensor.randomize(12345);  // Randomize tensor's data with a seed of 12345
         * ```
         * @endcode
         */
        void randomize(unsigned long long seed = 0) const;

        /**
         * @brief Clears the tensor's data by setting all elements to zero.
         *
         * This function resets the tensor's data to zero by filling the memory allocated for the tensor's data
         * with zero values. It uses the `cudaMemset` function to set all the values in the tensor's GPU memory to zero.
         * This is commonly used to clear or reset the tensor before using it for new computations.
         *
         * @note
         * - This function does not deallocate the memory; it only sets the values in the tensor's data to zero.
         * - The tensor's data memory is assumed to be allocated before calling this function. This is automatically managed
         *   when the tensor is created, so no additional memory allocation is needed.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a tensor with shape 2x3
         * tensor.clear();         // Clear the tensor's data by setting all elements to zero
         * ```
         * @endcode
         */
        void clear() const;

        /**
         * @brief Fills the tensor's data with a specified value.
         *
         * This function sets all elements in the tensor's data to the specified value.
         * It uses the `cudaMemset` function to fill the GPU memory allocated for the tensor with
         * the provided value. This is commonly used to initialize a tensor with a constant value.
         *
         * @param value The value to which all elements of the tensor will be set. This value is
         *              copied to every element in the tensor's data.
         *
         * @note
         * - This function does not deallocate the memory; it only sets the values in the tensor's data to the specified value.
         * - The tensor's data memory is assumed to be allocated before calling this function. This is automatically managed
         *   when the tensor is created, so no additional memory allocation is needed.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a tensor with shape 2x3
         * tensor.fill(5.0f);      // Fill the tensor's data with the value 5.0f
         * ```
         * @endcode
         */
        void fill(const value_type value) const;

        /**
         * @brief Fills the tensor's gradient data with a specified value.
         *
         * This function sets all elements in the tensor's gradient data to the specified value.
         * It uses the `cudaMemset` function to fill the GPU memory allocated for the gradient with
         * the provided value. This is typically used to initialize or reset the gradients before or
         * after backpropagation in neural network training.
         *
         * @param value The value to which all elements of the tensor's gradient will be set. This value is
         *              copied to every element in the tensor's gradient data.
         *
         * @note
         * - This function does not deallocate the gradient memory; it only sets the values in the tensor's gradient data to the specified value.
         * - The tensor must have been created with `requires_grad` set to `true`, otherwise the gradient memory will not be allocated, and calling this function will not have any effect.
         *
         * @code
         * Tensor tensor({2, 3}, true);  // Create a tensor with gradient support
         * tensor.fillGrad(0.0f);        // Set all gradient values to 0.0f
         * @endcode
         */
        void fillGrad(const value_type value) const;

        /**
         * @brief Reshapes the tensor to the specified shape.
         *
         * This function changes the shape of the tensor, adjusting the layout of the data
         * in memory. If the new shape has more elements than the current shape, the extra
         * elements will be initialized to zero. If the new shape has fewer elements, the
         * excess elements will be discarded.
         *
         * @param shape A `shape_type` (alias for `std::vector<int>`) representing the new dimensions of the tensor.
         *              The total number of elements in the new shape can be larger or smaller than the current shape.
         *
         * This function performs the following steps:
         * 1. It updates the tensor's shape to the new dimensions.
         * 2. If the new shape requires more elements than the original shape, the new elements are initialized to zero.
         * 3. If the new shape requires fewer elements, the excess data is discarded.
         *
         * @note
         * - This function does not reallocate memory. It simply adjusts how the existing data is interpreted based on the new shape.
         * - If the new shape has more elements than the current tensor, the excess elements will be initialized to zero.
         * - If the new shape has fewer elements, data beyond the new size will be discarded.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a tensor with shape 2x3
         * tensor.reshape(std::vector<int>({3, 2}));  // Reshape the tensor to shape 3x2, unused elements will be filled with zeros
         * ```
         * @endcode
         */
        void reshape(const shape_type& shape);

        /**
         * @brief Reshapes the tensor to the specified shape using an initializer list.
         *
         * This function changes the shape of the tensor, adjusting the layout of the data
         * in memory. The new shape is specified using an initializer list. If the new shape
         * has more elements than the current tensor, the extra elements will be initialized to zero.
         * If the new shape has fewer elements, the excess elements will be discarded.
         *
         * @param shape An `std::initializer_list<int>` representing the new dimensions of the tensor.
         *              The total number of elements in the new shape can be larger or smaller than the current shape.
         *
         * This function performs the following steps:
         * 1. It updates the tensor's shape to the new dimensions.
         * 2. If the new shape requires more elements than the current shape, the new elements are initialized to zero.
         * 3. If the new shape requires fewer elements, the excess data is discarded.
         *
         * @note
         * - This function does not reallocate memory. It simply adjusts how the existing data is interpreted based on the new shape.
         * - If the new shape has more elements than the current tensor, the excess elements will be initialized to zero.
         * - If the new shape has fewer elements, data beyond the new size will be discarded.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a tensor with shape 2x3
         * tensor.reshape({3, 2});  // Reshape the tensor to shape 3x2, unused elements will be filled with zeros
         * ```
         * @endcode
         */
        void reshape(const std::initializer_list<int>& shape);

        /**
         * @brief Transposes the tensor by swapping its dimensions and rearranging the data.
         *
         * This function performs a transpose on the tensor by swapping its rows and columns.
         * For a 2D tensor (matrix), it swaps the first and second dimensions, effectively
         * turning the rows into columns and vice versa. The tensor's data is rearranged using
         * a temporary buffer, and the shape is updated accordingly. The data is first copied to
         * a temporary memory space, then a CUDA kernel is used to perform the transposition.
         *
         * @note
         * - This function involves memory allocation and data copying. It creates a temporary
         *   tensor in GPU memory to hold the transposed data.
         * - After the transposition, the tensor's shape is updated, and the temporary buffer is freed.
         * - The function does not modify the original tensor's data but instead reinterprets
         *   the data with the new shape.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a tensor with shape 2x3
         * tensor.transpose();     // Transpose the tensor to shape 3x2
         * ```
         * @endcode
         */
        void transpose();

        /**
         * @brief Sets a specific element of the tensor's data to a given value.
         *
         * This function modifies a specific element of the tensor's data stored in GPU memory.
         * The element to be modified is specified by its position in the tensor's shape (given as a 2D index).
         * The function first copies the tensor's data from GPU memory to host memory, modifies the specified element,
         * and then copies the updated data back to the GPU memory.
         *
         * @param position A `shape_type` (alias for `std::vector<int>`) representing the 2D index (row, column)
         *                 of the element to modify.
         * @param value The value to which the specified element will be set.
         *
         * This function performs the following steps:
         * 1. It checks if the provided position is valid within the tensor's shape. If not, an exception is thrown.
         * 2. It copies the tensor's data from GPU memory to host memory using `cudaMemcpy`.
         * 3. It modifies the specified element at the given position in the tensor's data.
         * 4. It copies the updated data back to the GPU memory.
         *
         * @throws std::invalid_argument If the provided position is out of bounds.
         *
         * @note
         * - This function uses memory copying between host and device, which can introduce performance overhead.
         * - The tensor's data is modified on the host first and then copied back to the GPU. This approach may not be
         *   the most efficient for large tensors or frequent updates.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a tensor with shape 2x3
         * tensor.setData(std::vector<int>({1, 2}), 7.5f);  // Set the element at position (1, 2) to 7.5f
         * ```
         * @endcode
         */
        void setData(const shape_type& position, const value_type value) const;

        /**
         * @brief Sets a specific element of the tensor's data to a given value using an initializer list for position.
         *
         * This function modifies a specific element of the tensor's data stored in GPU memory.
         * The element to be modified is specified by its position in the tensor's shape, which is given as
         * an initializer list for the 2D index (row, column). The function copies the tensor's data from GPU memory
         * to host memory, modifies the specified element, and then copies the updated data back to GPU memory.
         *
         * @param position An `std::initializer_list<int>` representing the 2D index (row, column)
         *                 of the element to modify.
         * @param value The value to which the specified element will be set.
         *
         * This function performs the following steps:
         * 1. It checks if the provided position is valid within the tensor's shape. If not, an exception is thrown.
         * 2. It copies the tensor's data from GPU memory to host memory using `cudaMemcpy`.
         * 3. It modifies the specified element at the given position in the tensor's data.
         * 4. It copies the updated data back to the GPU memory.
         *
         * @throws std::invalid_argument If the provided position is out of bounds.
         *
         * @note
         * - This function uses memory copying between host and device, which can introduce performance overhead.
         * - The tensor's data is modified on the host first and then copied back to the GPU. This approach may not be
         *   the most efficient for large tensors or frequent updates.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});  // Create a tensor with shape 2x3
         * tensor.setData({1, 2}, 7.5f);  // Set the element at position (1, 2) to 7.5f
         * ```
         * @endcode
         */
        void setData(const std::initializer_list<int>& position, const value_type value) const;

        /// @}

        /// @name Math
        /// @{

        /**
         * @brief Adds two tensors element-wise and returns the result.
         *
         * This operator performs element-wise addition of two tensors and returns a new tensor
         * containing the sum of the corresponding elements from the two input tensors.
         *
         * @param other The tensor to be added to the current tensor.
         * @return A new tensor containing the element-wise sum of the two tensors.
         *
         * This function checks if the shapes of the two tensors match. If they do not, it throws
         * an `std::invalid_argument` exception. The function then creates a new tensor to hold the result
         * of the addition and uses a CUDA kernel (`MatrixAddKernel`) to compute the sum of the tensors' elements
         * in parallel on the GPU.
         *
         * @throws std::invalid_argument If the shapes of the two tensors do not match.
         *
         * @note
         * - The tensors must have the same shape. If they do not, an exception is thrown.
         * - This operator uses a CUDA kernel to perform the element-wise addition, and the result is
         *   stored in a new tensor, which is returned.
         * - The original tensors are not modified.
         *
         * @code
         * ```cpp
         * Tensor tensor1({2, 3});
         * Tensor tensor2({2, 3});
         * Tensor result = tensor1 + tensor2;  // Adds the two tensors element-wise
         * ```
         * @endcode
         */
        Tensor operator+(const Tensor& other) const;

        /**
         * @brief Subtracts one tensor from another element-wise and returns the result.
         *
         * This operator performs element-wise subtraction of two tensors and returns a new tensor
         * containing the result of subtracting the corresponding elements of the two input tensors.
         *
         * @param other The tensor to be subtracted from the current tensor.
         * @return A new tensor containing the element-wise difference of the two tensors.
         *
         * This function checks if the shapes of the two tensors match. If they do not, it throws
         * an `std::invalid_argument` exception. The function then creates a new tensor to hold the result
         * of the subtraction and uses a CUDA kernel (`MatrixSub`) to compute the element-wise subtraction
         * in parallel on the GPU.
         *
         * @throws std::invalid_argument If the shapes of the two tensors do not match.
         *
         * @note
         * - The tensors must have the same shape. If they do not, an exception is thrown.
         * - This operator uses a CUDA kernel to perform the element-wise subtraction, and the result is
         *   stored in a new tensor, which is returned.
         * - The original tensors are not modified.
         *
         * @code
         * ```cpp
         * Tensor tensor1({2, 3});
         * Tensor tensor2({2, 3});
         * Tensor result = tensor1 - tensor2;  // Subtracts tensor2 from tensor1 element-wise
         * ```
         * @endcode
         */
        Tensor operator-(const Tensor& other) const;

        /**
         * @brief Performs matrix multiplication of two tensors (matrices) and returns the result.
         *
         * This operator performs matrix multiplication between two tensors (2D matrices) and returns a new tensor
         * containing the result of the multiplication. The number of columns in the first tensor must match the number
         * of rows in the second tensor for matrix multiplication to be valid.
         *
         * @param other The tensor (matrix) to multiply with the current tensor.
         * @return A new tensor containing the result of the matrix multiplication.
         *
         * This function checks if the dimensions of the two tensors are compatible for matrix multiplication. If the
         * number of columns in the current tensor does not match the number of rows in the `other` tensor, it throws
         * an `std::invalid_argument` exception. It then creates a new tensor to hold the result of the multiplication
         * and uses a CUDA kernel (`GeneralMatrixMul`) to perform the matrix multiplication in parallel on the GPU.
         *
         * @throws std::invalid_argument If the matrix dimensions are incompatible for multiplication.
         *
         * @note
         * - The number of columns in the current tensor (`_shape[1]`) must match the number of rows in the `other` tensor
         *   (`other._shape[0]`) for the multiplication to be valid.
         * - This operator uses a CUDA kernel to perform matrix multiplication, and the result is stored in a new tensor,
         *   which is returned.
         * - The original tensors are not modified.
         *
         * @code
         * ```cpp
         * Tensor tensor1({2, 3});  // Create a 2x3 matrix
         * Tensor tensor2({3, 2});  // Create a 3x2 matrix
         * Tensor result = tensor1 * tensor2;  // Multiply the matrices (result will be 2x2)
         * ```
         * @endcode
         */
        Tensor operator*(const Tensor& other) const;

        /**
         * @brief Negates all elements of the tensor and returns the result.
         *
         * This operator performs element-wise negation of the tensor, returning a new tensor
         * that contains the negated values of the current tensor. Each element in the tensor is
         * multiplied by `-1` to compute its negation.
         *
         * @return A new tensor containing the element-wise negation of the current tensor.
         *
         * This function uses a CUDA kernel (`Negation`) to perform the negation of each element
         * in the tensor in parallel on the GPU. The result is stored in a new tensor, which is returned.
         *
         * @note
         * - This operator does not modify the original tensor; it returns a new tensor with the negated values.
         * - The operation is performed element-wise, meaning each individual element is negated.
         * - The operation utilizes GPU parallelization for efficiency.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});
         * Tensor result = -tensor;  // Negates all elements of the tensor
         * ```
         * @endcode
         */
        Tensor operator-() const;

        /**
         * @brief Computes the reciprocal (1/x) of each element in the tensor and updates the tensor in-place.
         *
         * This function computes the reciprocal (1/x) of each element in the tensor and stores the results
         * back into the original tensor. The operation is performed element-wise, where each element of the tensor
         * is replaced by its reciprocal.
         *
         * The function utilizes a temporary buffer allocated on the GPU to store the intermediate reciprocal values.
         * After the computation, the updated data is copied back to the original tensor in GPU memory.
         *
         * @note
         * - This operation is performed element-wise on the tensor's data.
         * - The original tensor is updated with the computed reciprocal values.
         * - The function uses GPU memory for efficient parallel computation.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});
         * tensor.recip();  // Computes the reciprocal of each element in the tensor
         * ```
         * @endcode
         */
        void recip() const;

        /// @}

        /// @name Printer
        /// @{

        /**
         * @brief Prints the gradient values of the tensor to an output stream.
         *
         * This function prints the gradient of the tensor (`_grad`) to the provided output stream (`os`).
         * The gradient data is first copied from GPU memory to host memory, and then it is printed in a 2D matrix
         * format where each row represents one dimension of the gradient. Each element in the gradient is printed,
         * separated by a space.
         *
         * @param os The output stream to which the gradient will be printed.
         * @return The same output stream (`os`), allowing for chaining of stream operations.
         *
         * This function performs the following steps:
         * 1. It allocates memory on the host and copies the gradient data from the device to the host.
         * 2. It uses `std::copy` to print the gradient values in a matrix format (row by row).
         * 3. The function prints each row of the gradient, with each value separated by a space.
         *
         * @note
         * - This function assumes that the gradient data has already been allocated and is valid.
         * - The gradient is copied from device (GPU) memory to host (CPU) memory for printing, which can be inefficient
         *   for large tensors.
         * - The function prints each row of the gradient tensor, enclosed in square brackets, with the elements separated by spaces.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3}, true);  // Create a tensor with gradient support
         * std::cout << "Gradient: " << std::endl;
         * tensor.printGrad(std::cout);  // Print the gradient of the tensor
         * ```
         * @endcode
         */
        std::ostream& printGrad(std::ostream& os) const;

        /**
         * @brief Prints the tensor's data to the standard output.
         *
         * This function prints the elements of the tensor to the standard output (usually the console),
         * formatted as a 2D matrix. Each row of the tensor is printed on a new line, and each element in a row
         * is separated by a space.
         *
         * The tensor's data is copied from GPU memory to host memory for printing, which could introduce performance
         * overhead, especially for large tensors.
         *
         * @note
         * - This function does not modify the tensor. It only prints the current tensor's data.
         * - The tensor's data is copied from the device (GPU) to the host (CPU) memory using `cudaMemcpy` before printing.
         *
         * @code
         * ```cpp
         * Tensor tensor({2, 3});
         * tensor.print();  // Prints the tensor's data to the standard output in matrix format
         * ```
         * @endcode
         */
        void print() const noexcept;

        /// @}


    private:
        size_type _size;
        shape_type _shape;
        value_type* _data;
        value_type* _grad;
        bool _requires_grad;
    };
}

#endif //TENSOR_CUH

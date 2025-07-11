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
#include <iterator>
#include <stdexcept>
#include <vector>
#include "dl_export.cuh"
#include "Dimension.cuh"

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
     * tensor3.dataInject({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}); // Fill tensor3 with values
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
     * - Most of the methods in this class involve CUDA operations and may throw the nz::CudaException in certain cases.
     *
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
        using shape_type = Dimension;

        friend DL_API std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        friend DL_API std::istream& operator>>(std::istream& is, const Tensor& tensor);

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
        explicit Tensor(const shape_type& shape, bool requires_grad = false);

        /**
         * @brief Constructs a Tensor object with specified shape, data, gradient requirement, and data location.
         *
         * @param shape A reference to the shape of the tensor (host-to-device). The shape determines the size of the tensor.
         * @param data A pointer to the initial data of the tensor. The data can be either on the host or device depending on the `host` parameter.
         * @param requires_grad A boolean indicating whether the tensor requires gradient computation.
         * @param host A boolean indicating whether the data pointed to by `data` is on the host or device. If true, data is on the host; otherwise, it is on the device.
         *
         * @return None. This is a constructor.
         *
         * This constructor initializes a `Tensor` object. It first calculates the total size of the tensor based on the provided shape. Then, it allocates device memory for the tensor's data using `cudaMalloc`.
         *
         * Depending on the value of the `host` parameter, it copies the data from either the host or another device memory location to the newly allocated device memory using `cudaMemcpy`.
         *
         * If the `requires_grad` parameter is `true`, it also allocates device memory for the gradient data of the tensor. Otherwise, it sets the gradient pointer `_grad` to `nullptr`.
         *
         * For memory management, the constructor allocates device memory for the tensor's data and gradient (if required). The responsibility of freeing this memory lies with the destructor of the `Tensor` class.
         *
         * In terms of exception handling, this constructor does not explicitly catch any CUDA errors. If a CUDA operation fails (e.g., `cudaMalloc` or `cudaMemcpy`), it will likely lead to undefined behavior in subsequent operations. It is the caller's responsibility to check for CUDA errors using `cudaGetLastError` or other appropriate methods.
         *
         * This constructor is a fundamental part of the `Tensor` class as it initializes the object's internal state.
         *
         * @throws None explicitly, but CUDA operations may fail and return an error code.
         *
         * @note
         * - Ensure that the `data` pointer is valid and points to enough data to fill the tensor according to the specified shape.
         * - The CUDA runtime environment should be properly initialized before calling this constructor.
         * - This constructor has a time complexity of O(1) for memory allocation and O(n) for data copying, where n is the total number of elements in the tensor (`_size`).
         *
         * @code
         * ```cpp
         * shape_type shape = {2, 3};
         * value_type data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
         * Tensor tensor(shape, data, true, true);
         * ```
         * @endcode
         */
        explicit Tensor(const shape_type& shape, value_type* data, bool requires_grad = false, bool host = true);

        /**
         * @brief Constructs a Tensor object with a specified shape, initializer list data, and gradient requirement.
         *
         * @param shape A reference to the shape of the tensor (host-to-device). The shape determines the dimensions and total size of the tensor.
         * @param data A std::initializer_list containing the initial data for the tensor (host-to-device).
         * @param requires_grad A boolean indicating whether the tensor requires gradient computation.
         *
         * @return None. This is a constructor.
         *
         * This constructor initializes a Tensor object. First, it calculates the total size of the tensor based on the provided shape. It then checks if the size of the std::initializer_list is sufficient to fill the tensor. If not, it throws a std::invalid_argument exception.
         *
         * For memory management, it allocates device memory for the tensor's data using cudaMalloc. If the tensor requires gradient computation, it also allocates device memory for the gradient data; otherwise, it sets the gradient pointer to nullptr.
         *
         * A temporary host buffer is created to hold the data from the std::initializer_list. The data is copied from the initializer list to the host buffer and then transferred from the host buffer to the device memory using cudaMemcpy. After the transfer, the temporary host buffer is deleted to prevent memory leaks.
         *
         * Regarding exception handling, it throws a std::invalid_argument if the initializer list size is insufficient. Any CUDA errors during memory allocation or data transfer are not explicitly caught here, and it's the caller's responsibility to check for CUDA errors.
         *
         * This constructor is an important part of the Tensor class as it provides a convenient way to initialize a tensor with an initializer list.
         *
         * @throws std::invalid_argument If the size of the std::initializer_list is less than the size of the tensor.
         *
         * @note
         * - Ensure that the std::initializer_list contains enough elements to fill the tensor according to the specified shape.
         * - The CUDA runtime environment should be properly initialized before calling this constructor.
         * - The time complexity of this constructor is O(n), where n is the total number of elements in the tensor, due to the loop that copies data from the initializer list to the host buffer.
         *
         * @code
         * ```cpp
         * #include <vector>
         *
         * shape_type shape = {2, 3};
         * try {
         *     Tensor tensor(shape, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
         * } catch (const std::invalid_argument& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        explicit Tensor(const shape_type& shape, const std::initializer_list<value_type>& data,
                        bool requires_grad = false);

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
        Tensor(Tensor&& other) noexcept(false);

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
        Tensor& operator=(Tensor&& other) noexcept(false);

        /**
        * @brief Destructor for Tensor.
        *
        * Releases all GPU memory allocated for the tensor's data and gradient. Ensures
        * that no memory leaks occur during the lifetime of the Tensor object.
        */
        ~Tensor() noexcept(false);

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
        [[nodiscard]] bool requiresGrad() const noexcept;

        /**
        * @brief Retrieves the shape of the tensor.
        *
        * @return A `shape_type` (alias for `std::vector<int>`) representing the dimensions of the tensor.
        *
        * The shape provides information about the size of each dimension in the tensor. For example,
        * a tensor with shape `{2, 3}` represents a 2x3 matrix. The shape is defined during construction
        * or reshaping of the tensor.
        */
        [[nodiscard]] shape_type shape() const noexcept;

        /**
        * @brief Retrieves the total number of elements in the tensor.
        *
        * @return A `size_type` (alias for `unsigned long long`) representing the total number of elements.
        *
        * This function calculates the product of the dimensions in the tensor's shape. For example,
        * a tensor with shape `{2, 3}` will have a size of 6. This value is useful for memory allocation
        * and tensor operations.
        */
        [[nodiscard]] size_type size() const noexcept;

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
        void setRequiresGrad(bool requires_grad);

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
        [[nodiscard]] value_type* data() const noexcept;

        /**
         * @brief Retrieves the tensor data from the device to the host and returns it as a std::vector.
         *
         * This member function is used to transfer the tensor data from the device memory to the host memory.
         * It returns a `std::vector` containing the tensor data.
         *
         * @param None
         *
         * @return A `std::vector` of `Tensor::value_type` containing the tensor data. Memory flow: device - to - host.
         *
         * **Memory Management Strategy**:
         * - A temporary array `temp` of size `_size` is dynamically allocated on the host using `new`.
         * - After the data is copied from the device to the host using `cudaMemcpy`, a `std::vector` is constructed from the temporary array.
         * - The temporary array `temp` is then deleted using `delete[]` to avoid memory leaks.
         *
         * **Exception Handling Mechanism**:
         * - This function is marked as `noexcept`, meaning it does not throw any exceptions. However, if `cudaMemcpy` fails, it may lead to undefined behavior.
         *
         * **Relationship with Other Components**:
         * - Depends on the `syncData()` function to synchronize the data before the transfer.
         * - Relies on `cudaMemcpy` to transfer data from the device to the host.
         * - The internal member variables `_data` and `_size` are used to access the device data and its size.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`).
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function.
         *
         * @warning
         * - If `cudaMemcpy` fails, the behavior of this function is undefined. Error checking for `cudaMemcpy` is not performed in this function.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * std::vector<Tensor::value_type> hostData = tensor.hostData();
         * ```
         * @endcode
         */
        [[nodiscard]] std::vector<value_type> hostData() const noexcept;

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
        [[nodiscard]] value_type* grad() const;

        /**
         * @brief Retrieves the gradient data of the tensor from the device to the host and returns it as a std::vector.
         *
         * This member function transfers the gradient data of the tensor from the device memory to the host memory.
         * It returns a `std::vector` containing the gradient data of the tensor.
         *
         * @param None
         *
         * @return A `std::vector` of `Tensor::value_type` containing the gradient data. Memory flow: device - to - host.
         *
         * **Memory Management Strategy**:
         * - A temporary array `temp` of size `_size` is dynamically allocated on the host using `new`.
         * - After the gradient data is copied from the device to the host using `cudaMemcpy`, a `std::vector` is constructed from the temporary array.
         * - The temporary array `temp` is then deleted using `delete[]` to avoid memory leaks.
         *
         * **Exception Handling Mechanism**:
         * - Throws `std::runtime_error` if the tensor does not require gradients (`_requires_grad` is `false`).
         * - If `cudaMemcpy` fails, it may lead to undefined behavior as error - checking for `cudaMemcpy` is not performed in this function.
         *
         * **Relationship with Other Components**:
         * - Depends on the `syncGrad()` function to synchronize the gradient data before the transfer.
         * - Relies on `cudaMemcpy` to transfer the gradient data from the device to the host.
         * - The internal member variables `_grad` and `_size` are used to access the device gradient data and its size.
         *
         * @throws std::runtime_error When the tensor does not require gradients.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`).
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function.
         * - Ensure that the tensor requires gradients before calling this function to avoid exceptions.
         *
         * @warning
         * - If `cudaMemcpy` fails, the behavior of this function is undefined.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * try {
         *     std::vector<Tensor::value_type> hostGrad = tensor.hostGrad();
         * } catch (const std::runtime_error& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] std::vector<value_type> hostGrad() const;

        /**
         * @brief Injects data or gradient data into the tensor.
         *
         * @param data A pointer to the data to be injected (host-to-device).
         * @param grad A boolean indicating whether to inject gradient data.
         *
         * @return void
         *
         * This function is responsible for injecting data or gradient data into the tensor.
         * For memory management, it uses `cudaMemcpy` to copy data from the host to the device.
         * If the `grad` parameter is `true`, it tries to copy data to the gradient buffer (`_grad`).
         * If the tensor does not require gradients (`_requires_grad` is `false`), it throws an exception.
         * If the `grad` parameter is `false`, it copies data to the main data buffer (`_data`).
         *
         * The exception handling mechanism is in place to catch any CUDA memory copy errors.
         * If the `cudaMemcpy` operation fails, it throws a `nz::CudaException` with an appropriate error message.
         *
         * This function is closely related to the `Tensor` class as it modifies the internal data of the tensor.
         *
         * @throws nz::CudaException If the CUDA memory copy fails or if the tensor does not require gradients when trying to inject gradient data.
         *
         * @note
         * - The input data pointer `data` should point to a valid memory location with enough data to fill the tensor.
         * - Ensure that the CUDA environment is properly initialized before calling this function.
         *
         * @warning
         * This function is not safe.
         * If the length of the input array pointed to by `data` is less than the size of the tensor, it will lead to undefined behavior and potentially cause unknown issues in the program.
         *
         * @code
         * ```cpp
         * Tensor tensor({1, 3});
         * float data[] = {1.0, 2.0, 3.0};
         * try {
         *     tensor.dataInject(data, false);
         * } catch (const std::runtime_error& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        void dataInject(value_type* data, bool grad = false) const;

        /**
         * @brief Injects data or gradient data into the tensor using iterators.
         *
         * @param begin An iterator pointing to the beginning of the input data range (host-to-device).
         * @param end An iterator pointing to the end of the input data range (host-to-device).
         * @param grad A boolean indicating whether to inject gradient data. Defaults to false.
         *
         * @return void
         *
         * This function injects data or gradient data into the tensor using the provided iterator range.
         * First, it checks if the length of the input range (determined by `std::distance(begin, end)`) is at least as large as the size of the tensor (`_size`). If not, it throws a `std::runtime_error`.
         *
         * For memory management, it allocates a temporary host array `host_data` of size `_size` to store the data from the iterator range. The data is then copied from the iterator range to this temporary array. After that, it calls the `dataInject` function with the temporary array and the `grad` flag.
         *
         * In case of an exception during the call to the `dataInject` function, the temporary array is deleted to prevent memory leaks. Finally, the temporary array is deleted after the call to `dataInject` returns successfully.
         *
         * The exception handling mechanism catches any `nz::CudaException` or `std::runtime_error` thrown by the `dataInject` function and re - throws it after cleaning up the temporary memory.
         *
         * This function is closely related to the `Tensor` class and the other `dataInject` function as it uses the other `dataInject` function to perform the actual data injection.
         *
         * @throws std::runtime_error If the length of the input array is less than the size of the tensor.
         * @throws nz::CudaException If the CUDA memory copy fails or if the tensor does not require gradients when trying to inject gradient data.
         *
         * @note
         * - The iterators `begin` and `end` should be valid and form a proper range.
         * - The input data should be convertible to the `value_type` of the tensor.
         * - The time complexity of this function is O(n), where n is the size of the tensor (`_size`), due to the loop that copies data from the iterator range to the temporary array.
         *
         * @code
         * ```cpp
         * #include <vector>
         * Tensor tensor({1,3});
         * std::vector<float> data = {1.0f, 2.0f, 3.0f};
         * try {
         *     tensor.dataInject(data.begin(), data.end(), false);
         * } catch (const std::runtime_error& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        template <typename Iterator>
        void dataInject(Iterator begin, Iterator end, const bool grad = false) const {
            if (std::distance(begin, end) < _size) {
                throw std::runtime_error("The length of the input array is less than the size of the tensor.");
            }
            auto* host_data = new value_type[_size];
            auto it = begin;
            for (auto i = 0; i < _size; ++i, ++it) {
                host_data[i] = static_cast<value_type>(*it);
            }
            try {
                dataInject(host_data, grad);
            }
            catch ([[maybe_unused]] const std::runtime_error& e) {
                delete[] host_data;
                throw;
            }
            delete[] host_data;
        }

        /**
         * @brief Injects data or gradient data into the tensor using a std::initializer_list.
         *
         * @param data A std::initializer_list containing the data to be injected (host-to-device).
         * @param grad A boolean indicating whether to inject gradient data.
         *
         * @return void
         *
         * This function serves as a wrapper that calls another `dataInject` function, passing the begin and end iterators of the provided `std::initializer_list`.
         * In terms of memory management, it relies on the underlying `dataInject` function to handle memory operations for the actual data injection.
         * Regarding exception handling, it simply propagates any exceptions thrown by the underlying `dataInject` function without additional handling.
         * This function is closely related to the `Tensor` class and the other `dataInject` functions as it leverages the existing data injection logic.
         *
         * @throws std::runtime_error If the length of the input array is less than the size of the tensor.
         * @throws nz::CudaException If the CUDA memory copy fails or if the tensor does not require gradients when trying to inject gradient data.
         *
         * @note
         * - The `std::initializer_list` should contain enough elements to fill the tensor.
         * - This function has a time complexity of O(1) for the wrapper itself, but the overall complexity depends on the underlying `dataInject` function which is O(n) where n is the size of the tensor.
         *
         * @code
         * ```cpp
         * Tensor tensor({1,3});
         * try {
         *     tensor.dataInject({1.0f, 2.0f, 3.0f}, false);
         * } catch (const std::runtime_error& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        void dataInject(const std::initializer_list<value_type>& data, bool grad = false) const;

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
        void zeroGrad() const;

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
         * @param isGrad A boolean flag indicating whether to fill the gradients or the data. If true, gradients are filled; otherwise, data is filled (host-to-device).
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
        void fill(value_type value, bool isGrad = false) const;

        /**
         * @brief Fill a specific matrix slice within the Tensor with a given value.
         *
         * This method allows users to populate a particular matrix slice of the Tensor (specified by batch and channels)
         * with a provided value. It also supports filling the gradient matrix if the Tensor requires gradient computation
         * and the `isGrad` flag is set.
         *
         * @param value The value used to fill the matrix slice. Memory flow: host-to-function, passed from the calling code.
         * @param batch The index of the batch. Memory flow: host-to-function, passed from the calling code.
         * @param channels The index of the channels. Memory flow: host-to-function, passed from the calling code.
         * @param isGrad A boolean indicating whether to fill the gradient matrix. Memory flow: host-to-function, passed from the calling code.
         *
         * @return None
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any additional memory. It operates on the existing `_data` or `_grad` buffer of the Tensor.
         *
         * **Exception Handling Mechanism**:
         * - Throws `std::invalid_argument` if the provided `batch` or `channels` indices are out of bounds.
         * - Throws `std::invalid_argument` if `isGrad` is `true` but the Tensor does not require gradient computation.
         *
         * **Relationship with Other Components**:
         * - Depends on the `_shape` object to access tensor shape information and calculate offsets.
         * - Relies on the `krnl::Fill` CUDA kernel to perform the actual filling operation.
         *
         * @throws std::invalid_argument When `batch` or `channels` are out of bounds or when trying to fill gradients of a non - gradient - requiring Tensor.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the matrix slice (`_shape[2] * _shape[3]`).
         * - Ensure that the `krnl::Fill` CUDA kernel is properly implemented and the CUDA environment is set up correctly.
         * - Verify that the `_shape` object provides accurate shape and stride information.
         *
         * @warning
         * - Incorrect CUDA kernel usage may lead to runtime errors or undefined behavior.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * Tensor::value_type fillValue = 2.0;
         * Tensor::size_type batchIndex = 0;
         * Tensor::size_type channelIndex = 1;
         * bool fillGrad = false;
         * try {
         *     tensor.fillMatrix(fillValue, batchIndex, channelIndex, fillGrad);
         * } catch (const std::invalid_argument& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        void fillMatrix(value_type value, size_type batch, size_type channels, bool isGrad = false);


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
         * @brief Sets the value of an element in the tensor or its gradient at a specified position.
         *
         * This member function allows you to set the value of a specific element in the tensor or its gradient.
         * It first validates the position and the gradient setting based on the tensor's requirements.
         *
         * @param position The position in the tensor where the value will be set. Memory location: host - to - device.
         * @param value The value to be set at the specified position. Memory location: host - to - device.
         * @param isGrad A boolean indicating whether to set the value in the gradient or the tensor data. Memory location: host - to - device.
         *
         * @return None
         *
         * **Memory Management Strategy**:
         * - A temporary array `data` of size `_size` is allocated on the host using `malloc`.
         * - The data from the device (either tensor data or gradient) is copied to the host using `cuStrm::StreamManager<value_type>::Instance().memcpy`.
         * - After the value is set at the specified position in the host - side data, the updated data is copied back to the device.
         * - The temporary array `data` is freed using `free` to avoid memory leaks.
         *
         * **Exception Handling Mechanism**:
         * - Throws `std::invalid_argument` if the `position` is out of bounds of the tensor's shape.
         * - Throws `std::invalid_argument` if `isGrad` is `true` but the tensor does not require gradients.
         * - If any of the `cuStrm::StreamManager` operations fail, it may lead to undefined behavior as error - checking is not explicitly done in this function.
         *
         * **Relationship with Other Components**:
         * - Depends on `cuStrm::StreamManager<value_type>::Instance()` for memory copying and data synchronization operations.
         * - Relies on the `_shape` member variable to validate the position and calculate the index in the data array.
         * - Uses the `_data` and `_grad` member variables to access the tensor data and its gradient.
         *
         * @throws std::invalid_argument When the position is out of bounds or when trying to set the gradient of a tensor that does not require gradients.
         *
         * @note
         * - The time complexity of this function is O(n) due to the memory copying operations, where n is the number of elements in the tensor (`_size`).
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function.
         * - Ensure that the `position` is within the valid range of the tensor's shape to avoid exceptions.
         * - If setting the gradient, ensure that the tensor requires gradients.
         *
         * @warning
         * - If any of the `cuStrm::StreamManager` operations fail, the behavior of this function is undefined.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * Tensor::shape_type position = {0, 0, 0, 0};
         * Tensor::value_type value = 1.0;
         * bool isGrad = false;
         * try {
         *     tensor.setData(position, value, isGrad);
         * } catch (const std::invalid_argument& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        void setData(const shape_type& position, value_type value, bool isGrad = false) const;

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
         * @brief Performs element-wise division between two Tensors.
         *
         * This function overloads the division operator to perform element-wise division between the current Tensor and another Tensor.
         * It broadcasts the shapes of the two Tensors if necessary and creates a new Tensor to store the result.
         *
         * @param other The Tensor to divide the current Tensor by. Memory flow: host-to-function, as the object is passed from the calling code to the function.
         *
         * @return A new Tensor containing the result of the element-wise division. Memory flow: function-to-host, as the result is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - A new Tensor object `result` is created within the function to store the result of the division. The memory for this Tensor is managed automatically by its constructor and destructor.
         *
         * **Exception Handling Mechanism**:
         * - This function does not explicitly throw exceptions. However, the `_shape.Broadcast` method or the `tensorElementwiseDivide` function may throw exceptions if there are issues with shape broadcasting or the division operation.
         *
         * **Relationship with Other Components**:
         * - This function depends on the `_shape.Broadcast` method to handle shape broadcasting between the two Tensors.
         * - It also relies on the `tensorElementwiseDivide` function to perform the actual element-wise division operation.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the resulting Tensor after broadcasting. This is because the `tensorElementwiseDivide` function needs to process each element.
         * - Ensure that the `_shape.Broadcast` method and the `tensorElementwiseDivide` function are correctly implemented.
         *
         * @warning
         * - Division by zero may occur if the `other` Tensor contains zero elements, which can lead to undefined behavior.
         *
         * @code
         * ```cpp
         * Tensor tensor1;
         * Tensor tensor2;
         * Tensor result = tensor1 / tensor2;
         * ```
         * @endcode
         */
        Tensor operator/(const Tensor& other) const;

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
         * @brief Checks if two Tensor objects are equal.
         *
         * @param other The other Tensor object to compare with. Memory flow: device - to - host (data is copied from device to host for comparison).
         *
         * @return Returns true if the two Tensor objects are equal, false otherwise.
         *
         * This function compares two Tensor objects for equality. First, it checks if the `_requires_grad` flags of the two Tensors are the same. If they differ, the function immediately returns false. Then, it compares the shapes of the two Tensors. If the shapes are not equal, the function also returns false.
         *
         * After that, it allocates host memory for temporary storage of the data from the device memory of both Tensors. It copies the data from the device to the host and compares each element one by one. If any element in the data differs, it frees the allocated host memory and returns false.
         *
         * If the `_requires_grad` flag is set to true, it repeats the same process for the gradients of the Tensors. If any element in the gradients differs, it frees the allocated host memory and returns false.
         *
         * Finally, if all comparisons pass, it frees the allocated host memory and returns true.
         *
         * **Memory Management Strategy**:
         * - Two arrays `temp` and `temp_other` of size `_size` are dynamically allocated on the host using `new[]`. They are freed using `delete[]` either when a difference is found or at the end of the function.
         *
         * **Exception Handling Mechanism**:
         * - The CUDA memory copy operations (`cudaMemcpy`) may return error codes indicating failures. It is assumed that the calling code or the CUDA runtime will handle these errors appropriately.
         *
         * **Relationship with Other Components**:
         * - Depends on the `_requires_grad`, `_shape`, `_size`, `_data`, and `_grad` members of the `Tensor` class.
         * - Uses CUDA memory copy operations (`cudaMemcpy`) to transfer data from device to host.
         *
         * @note
         * - Be aware of potential CUDA errors during memory copy operations and handle them appropriately in the calling code.
         * - The function has a time complexity of O(n), where n is the number of elements in the Tensor, due to the element - by - element comparison.
         *
         * @code
         * ```cpp
         * Tensor tensor1; // Assume Tensor1 is properly initialized
         * Tensor tensor2; // Assume Tensor2 is properly initialized
         * bool isEqual = tensor1 == tensor2;
         * ```
         * @endcode
         */
        bool operator==(const Tensor& other) const;

        /**
         * @brief Checks if two Tensor objects are not equal.
         *
         * @param other The other Tensor object to compare with. Memory flow: device - to - host (the comparison in the `operator==` function may involve data transfer from device to host).
         *
         * @return Returns true if the two Tensor objects are not equal, false otherwise.
         *
         * This function checks the inequality of two Tensor objects. It simply negates the result of the `operator==` function. So, it relies on the implementation of the `operator==` to determine the equality of the two Tensors.
         *
         * **Memory Management Strategy**:
         * - All memory management related to the comparison is handled by the `operator==` function. This function itself does not allocate or free any memory.
         *
         * **Exception Handling Mechanism**:
         * - Any exceptions that may occur during the comparison are handled by the `operator==` function. This function does not have its own exception handling mechanism.
         *
         * **Relationship with Other Components**:
         * - Depends entirely on the `operator==` function of the `Tensor` class.
         *
         * @note
         * - The time complexity of this function is the same as that of the `operator==` function, which is O(n) where n is the number of elements in the Tensor.
         *
         * @code
         * ```cpp
         * Tensor tensor1; // Assume Tensor1 is properly initialized
         * Tensor tensor2; // Assume Tensor2 is properly initialized
         * bool isNotEqual = tensor1 != tensor2;
         * ```
         * @endcode
         */
        bool operator!=(const Tensor& other) const;

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

        /**
         * @brief Compute the sum of all elements in the Tensor.
         *
         * @return The sum of all elements in the Tensor as a value of type `Tensor::value_type`.
         *
         * This function calculates the sum of all elements in the Tensor using CUDA parallel processing. It first determines the block and grid dimensions for the CUDA kernel. Then, it allocates device memory for intermediate results and host memory to store the results copied from the device. The `krnl::Summation` CUDA kernel is launched to perform partial sums on the device. After the kernel execution, the partial sums are copied from the device to the host using `cudaMemcpy`. Finally, the partial sums on the host are added together to obtain the total sum, and the allocated host and device memory are freed.
         *
         * Memory management:
         * - Host memory is allocated for `hData` using `new[]` and freed using `delete[]`.
         * - Device memory is allocated for `dData` using `cudaMalloc` and freed using `cudaFree`.
         *
         * Exception handling:
         * - The `CHECK` macro is used to handle CUDA API errors. If a CUDA API call fails, the `CHECK` macro will throw an exception, and the function will terminate.
         *
         * Relationship with other components:
         * - This function depends on the `krnl::Summation` CUDA kernel to perform partial sums on the device.
         * - It also depends on the `CHECK` macro to handle CUDA API errors.
         *
         * @throws [Exception type thrown by CHECK macro] If there are CUDA API errors during memory allocation, kernel execution, or memory copying.
         *
         * @note
         * - The time complexity of this function is approximately O(n), where n is the number of elements in the Tensor (`_size`). The CUDA kernel parallelizes the partial sum calculation, and the final sum on the host is a linear operation over the number of grid blocks.
         * - Ensure that the CUDA device is properly initialized before calling this function.
         *
         * @code
         * ```cpp
         * nz::data::Tensor tensor({2, 3}, true);
         * // Assume tensor is filled with some values
         * nz::data::Tensor::value_type sum_result = tensor.sum();
         * ```
         * @endcode
         */
        [[nodiscard]] value_type sum() const;

        /**
         * @brief Computes the sum of elements in a specific batch and channel of a Tensor.
         *
         * @param batch The batch index. This value should be within the valid range of the Tensor's batch dimension. Memory flow: host - to - device (used for index calculation on the host side).
         * @param channel The channel index. This value should be within the valid range of the Tensor's channel dimension. Memory flow: host - to - device (used for index calculation on the host side).
         *
         * @return The sum of elements in the specified batch and channel of the Tensor.
         *
         * This function calculates the sum of elements in a particular batch and channel of a Tensor. First, it checks if the provided `batch` and `channel` indices are valid. If not, it throws a `std::invalid_argument` exception. Then, it calculates the size of the region to be summed based on the Tensor's shape. It allocates device memory for intermediate results and host memory to receive the intermediate results from the device. It determines the offset in the Tensor's data based on the `batch` and `channel` indices. The `krnl::Summation` kernel is then launched to perform the partial summation on the device. After that, the intermediate results are copied from the device to the host. Finally, the function sums up all the intermediate results on the host, frees the allocated host and device memory, and returns the final sum.
         *
         * **Memory Management Strategy**:
         * - On the host side, an array `hData` of size `grid.x` is dynamically allocated using `new[]` and later freed using `delete[]`.
         * - On the device side, memory for `dData` is allocated using `cuStrm::StreamManager<value_type>::Instance().malloc` and freed using `cuStrm::StreamManager<value_type>::Instance().free`.
         *
         * **Exception Handling Mechanism**:
         * - Throws a `std::invalid_argument` exception if the provided `batch` or `channel` indices are out of the valid range of the Tensor's shape.
         * - The CUDA memory allocation, copying, and kernel launch operations may return error codes indicating failures. It is assumed that the calling code or the CUDA runtime will handle these errors appropriately.
         *
         * **Relationship with Other Components**:
         * - Depends on the `_shape` member of the `Tensor` class to get the shape information and strides.
         * - Uses the `krnl::Summation` kernel to perform the partial summation on the device.
         * - Relies on `cuStrm::StreamManager<value_type>::Instance()` for CUDA memory management (malloc, memcpy, free) operations.
         *
         * @throws std::invalid_argument If the provided `batch` or `channel` indices are out of the valid range of the Tensor's shape.
         *
         * @note
         * - Ensure that the provided `batch` and `channel` indices are within the valid range of the Tensor's shape to avoid exceptions.
         * - The CUDA operations such as memory allocation, copying, and kernel launch have their own error handling mechanisms. The calling code should be prepared to handle potential CUDA errors.
         *
         * @code
         * ```cpp
         * Tensor tensor; // Assume Tensor is properly initialized
         * Tensor::size_type batch = 0;
         * Tensor::size_type channel = 1;
         * Tensor::value_type sumResult = tensor.sum(batch, channel);
         * ```
         * @endcode
         */
        [[nodiscard]] value_type sum(size_type batch, size_type channel) const;

        /**
         * @brief Finds the maximum value in the tensor.
         *
         * This member function retrieves the tensor data from the device to the host and then iterates through it to find the maximum value.
         *
         * @param None
         *
         * @return The maximum value of type `Tensor::value_type` in the tensor. Memory flow: device - to - host.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), due to the linear traversal of the tensor data.
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function, as it depends on `hostData()`.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * try {
         *     Tensor::value_type maxVal = tensor.max();
         *     std::cout << "The maximum value in the tensor is: " << maxVal << std::endl;
         * } catch (const std::exception& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] value_type max() const;

        /**
         * @brief Finds the maximum value in a specific batch and channel of the tensor.
         *
         * This member function first validates the provided batch and channel indices.
         * If they are valid, it calculates the offset in the tensor data and then finds the maximum value within that subset of the tensor.
         *
         * @param batch The batch index. Memory location: host - to - device (used for index calculation).
         * @param channel The channel index. Memory location: host - to - device (used for index calculation).
         *
         * @return The maximum value of type `Tensor::value_type` in the specified batch and channel of the tensor. Memory flow: device - to - host.
         *
         * @throws std::invalid_argument When the `batch` or `channel` index is out of bounds.
         *
         * @note
         * - The time complexity of this function is O(m), where m is the number of elements in the specified batch and channel (`_shape[2] * _shape[3]`), due to the linear traversal of the subset of the tensor data.
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function, as it depends on `hostData()`.
         * - Ensure that the `batch` and `channel` indices are within the valid range of the tensor's shape to avoid exceptions.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * Tensor::size_type batch = 0;
         * Tensor::size_type channel = 0;
         * try {
         *     Tensor::value_type maxVal = tensor.max(batch, channel);
         *     std::cout << "The maximum value in batch " << batch << " and channel " << channel << " is: " << maxVal << std::endl;
         * } catch (const std::exception& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] value_type max(size_type batch, size_type channel) const;

        /**
         * @brief Finds the minimum value in the entire tensor.
         *
         * This function retrieves the tensor data from the device to the host and iterates through it to determine the minimum value.
         *
         * @param None
         *
         * @return The minimum value of type `Tensor::value_type` in the tensor. Memory flow: device - to - host.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), due to the linear traversal of the tensor data.
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function, as it depends on `hostData()`.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * try {
         *     Tensor::value_type minVal = tensor.min();
         *     std::cout << "The minimum value in the tensor is: " << minVal << std::endl;
         * } catch (const std::exception& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] value_type min() const;

        /**
         * @brief Finds the minimum value in a specific batch and channel of the tensor.
         *
         * This function first validates the provided batch and channel indices. If they are valid, it calculates the offset in the tensor data and then finds the minimum value within that subset of the tensor.
         *
         * @param batch The batch index. Memory location: host - to - device (used for index calculation).
         * @param channel The channel index. Memory location: host - to - device (used for index calculation).
         *
         * @return The minimum value of type `Tensor::value_type` in the specified batch and channel of the tensor. Memory flow: device - to - host.
         *
         * @throws std::invalid_argument When the `batch` or `channel` index is out of bounds.
         *
         * @note
         * - The time complexity of this function is O(m), where m is the number of elements in the specified batch and channel (`_shape[2] * _shape[3]`), due to the linear traversal of the subset of the tensor data.
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function, as it depends on `hostData()`.
         * - Ensure that the `batch` and `channel` indices are within the valid range of the tensor's shape to avoid exceptions.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * Tensor::size_type batch = 0;
         * Tensor::size_type channel = 0;
         * try {
         *     Tensor::value_type minVal = tensor.min(batch, channel);
         *     std::cout << "The minimum value in batch " << batch << " and channel " << channel << " is: " << minVal << std::endl;
         * } catch (const std::exception& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] value_type min(size_type batch, size_type channel) const;

        /**
         * @brief Finds the first occurrence of a given value in the entire tensor and returns its shape indices.
         *
         * This function retrieves the tensor data from the device to the host, then iterates through the data to find the first element equal to the given value.
         * Once found, it calculates the corresponding shape indices (batch, channel, height, width) and returns them.
         *
         * @param value The value to search for in the tensor. Memory location: host - to - device (used for comparison).
         *
         * @return A `Tensor::shape_type` object representing the shape indices (batch, channel, height, width) of the first occurrence of the given value in the tensor. Memory flow: device - to - host.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), due to the linear traversal of the tensor data.
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function, as it depends on `hostData()`.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * Tensor::value_type targetValue = 5.0;
         * try {
         *     Tensor::shape_type indices = tensor.find(targetValue);
         *     std::cout << "The first occurrence of " << targetValue << " is at indices: ("
         *               << indices[0] << ", " << indices[1] << ", " << indices[2] << ", " << indices[3] << ")" << std::endl;
         * } catch (const std::exception& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] shape_type find(value_type value) const;

        /**
         * @brief Finds the first occurrence of a given value in a specific batch and channel of the tensor and returns its shape indices.
         *
         * This function first calculates the offset in the tensor data based on the provided batch and channel indices.
         * It then retrieves the tensor data from the device to the host and iterates through the subset of data in the specified batch and channel to find the first element equal to the given value.
         * Once found, it calculates the height and width indices and returns the complete shape indices (batch, channel, height, width).
         *
         * @param value The value to search for in the tensor. Memory location: host - to - device (used for comparison).
         * @param batch The batch index. Memory location: host - to - device (used for index calculation).
         * @param channel The channel index. Memory location: host - to - device (used for index calculation).
         *
         * @return A `Tensor::shape_type` object representing the shape indices (batch, channel, height, width) of the first occurrence of the given value in the specified batch and channel of the tensor. Memory flow: device - to - host.
         *
         * @throws std::invalid_argument When the `batch` or `channel` index is out of bounds.
         *
         * @note
         * - The time complexity of this function is O(m), where m is the number of elements in the specified batch and channel (`_shape[2] * _shape[3]`), due to the linear traversal of the subset of the tensor data.
         * - Ensure that the CUDA runtime environment is properly initialized and the device memory is valid before calling this function, as it depends on `hostData()`.
         * - Ensure that the `batch` and `channel` indices are within the valid range of the tensor's shape to avoid unexpected behavior.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * Tensor::value_type targetValue = 5.0;
         * Tensor::size_type batch = 0;
         * Tensor::size_type channel = 0;
         * try {
         *     Tensor::shape_type indices = tensor.find(targetValue, batch, channel);
         *     std::cout << "The first occurrence of " << targetValue << " in batch " << batch << " and channel " << channel
         *               << " is at indices: (" << indices[0] << ", " << indices[1] << ", " << indices[2] << ", " << indices[3] << ")" << std::endl;
         * } catch (const std::exception& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] shape_type find(value_type value, size_type batch, size_type channel) const;

        /**
         * @brief Compute the sum of the exponential values of all elements in the Tensor.
         *
         * @return The sum of the exponential values of all elements in the Tensor as a value of type `Tensor::value_type`.
         *
         * This function calculates the sum of the exponential values of all elements in the Tensor. It first configures the CUDA block and grid dimensions. Then, it allocates device memory for intermediate results and host memory to hold the copied results from the device. The `krnl::SummationExp` CUDA kernel is launched to compute the partial sums of the exponential values on the device. After the kernel execution, the partial sums are transferred from the device to the host using `cudaMemcpy`. Finally, the partial sums on the host are added together to obtain the total sum, and the allocated host and device memory are freed.
         *
         * Memory management:
         * - Host memory is allocated for `hData` using `new[]` and freed using `delete[]`.
         * - Device memory is allocated for `dData` using `cudaMalloc` and freed using `cudaFree`.
         *
         * Exception handling:
         * - The `CHECK` macro is used to handle CUDA API errors. If a CUDA API call fails, the `CHECK` macro will throw an exception, and the function will terminate.
         *
         * Relationship with other components:
         * - This function depends on the `krnl::SummationExp` CUDA kernel to perform the partial sums of exponential values on the device.
         * - It also depends on the `CHECK` macro to handle CUDA API errors.
         *
         * @throws [Exception type thrown by CHECK macro] If there are CUDA API errors during memory allocation, kernel execution, or memory copying.
         *
         * @note
         * - The time complexity of this function is approximately O(n), where n is the number of elements in the Tensor (`_size`). The CUDA kernel parallelizes the partial sum calculation of exponential values, and the final sum on the host is a linear operation over the number of grid blocks.
         * - Ensure that the CUDA device is properly initialized before calling this function.
         *
         * @code
         * ```cpp
         * nz::data::Tensor tensor({2, 3}, true);
         * // Assume tensor is filled with some values
         * nz::data::Tensor::value_type exp_sum_result = tensor.expSum();
         * ```
         * @endcode
         */
        [[nodiscard]] value_type expSum() const;

        /**
         * @brief Computes the sum of exponential values of elements in a specific batch and channel of a Tensor.
         *
         * @param batch The batch index. Memory flow: host - to - device (used for index calculation on the host side).
         * @param channel The channel index. Memory flow: host - to - device (used for index calculation on the host side).
         *
         * @return The sum of exponential values of elements in the specified batch and channel of the Tensor.
         *
         * This function calculates the sum of the exponential values of elements within a particular batch and channel of a Tensor. First, it validates the provided `batch` and `channel` indices. If they are out of the valid range of the Tensor's shape, it throws a `std::invalid_argument` exception.
         *
         * After validation, it computes the size of the region to be processed based on the Tensor's shape. It then allocates device memory for intermediate results (`dData`) and host memory (`hData`) to receive the computed values from the device. The offset in the Tensor's data is determined according to the `batch` and `channel` indices.
         *
         * The `krnl::SummationExp` kernel is launched to compute the exponential of each element and perform partial summation on the device. The intermediate results are then copied from the device to the host. Finally, the function sums up all the intermediate results on the host, frees the allocated host and device memory, and returns the final sum.
         *
         * **Memory Management Strategy**:
         * - On the host side, an array `hData` of size `grid.x` is dynamically allocated using `new[]` and later freed using `delete[]`.
         * - On the device side, memory for `dData` is allocated using `cuStrm::StreamManager<value_type>::Instance().malloc` and freed using `cuStrm::StreamManager<value_type>::Instance().free`.
         *
         * **Exception Handling Mechanism**:
         * - Throws a `std::invalid_argument` exception if the provided `batch` or `channel` indices are out of the valid range of the Tensor's shape.
         * - The CUDA memory allocation, copying, and kernel launch operations may return error codes indicating failures. It is assumed that the calling code or the CUDA runtime will handle these errors appropriately.
         *
         * **Relationship with Other Components**:
         * - Depends on the `_shape` member of the `Tensor` class to get the shape information and strides.
         * - Uses the `krnl::SummationExp` kernel to perform the exponential calculation and partial summation on the device.
         * - Relies on `cuStrm::StreamManager<value_type>::Instance()` for CUDA memory management (malloc, memcpy, free) operations.
         *
         * @throws std::invalid_argument If the provided `batch` or `channel` indices are out of the valid range of the Tensor's shape.
         *
         * @note
         * - Ensure that the provided `batch` and `channel` indices are within the valid range of the Tensor's shape to avoid exceptions.
         * - Be aware of potential CUDA errors during memory allocation, copying, and kernel launch operations and handle them appropriately in the calling code.
         *
         * @code
         * ```cpp
         * Tensor tensor; // Assume Tensor is properly initialized
         * size_t batch = 0;
         * size_t channel = 1;
         * Tensor::value_type expSumResult = tensor.expSum(batch, channel);
         * ```
         * @endcode
         */
        [[nodiscard]] value_type expSum(size_t batch, size_t channel) const;

        /**
         * @brief Synchronize the tensor data by waiting for all CUDA stream write operations to complete.
         *
         * This function accesses the singleton instance of `cuStrm::StreamManager` specialized for the `value_type` of the `Tensor` class. It then calls the `syncData` method of this instance, passing the `_data` member of the `Tensor` object. This operation blocks the host until all CUDA stream write operations on the `_data` are finished.
         *
         * @param None
         *
         * @return None
         *
         * Memory management for the `_data` is assumed to be handled elsewhere in the codebase. There is no memory allocation or deallocation within this function.
         * This function does not have an explicit exception - handling mechanism. It relies on the `syncData` method of the `cuStrm::StreamManager` instance to manage any errors that may occur during the synchronization process.
         *
         * @note
         * - The time complexity of this function depends on the time required for the CUDA stream write operations on `_data` to complete. In the worst - case scenario, if there are long - running write operations, it may take a significant amount of time.
         *
         * @code
         * ```cpp
         * // Assume Tensor is defined and an instance is created
         * Tensor tensor;
         * tensor.syncData();
         * ```
         * @endcode
         */
        void syncData() const;

        /**
         * @brief Synchronize the gradient data of the tensor if gradient computation is required.
         *
         * This function first checks the `_requires_grad` flag of the `Tensor` object. If the flag is set to `true`, it accesses the singleton instance of `cuStrm::StreamManager` specialized for the `value_type` of the `Tensor` class. Then it calls the `syncData` method of this instance, passing the `_grad` member of the `Tensor` object. This operation blocks the host until all CUDA stream write operations on the `_grad` are completed.
         *
         * @param None
         *
         * @return None
         *
         * Memory management for the `_grad` is assumed to be handled elsewhere in the codebase. There is no memory allocation or deallocation within this function.
         * This function does not have an explicit exception - handling mechanism. It relies on the `syncData` method of the `cuStrm::StreamManager` instance to manage any errors that may occur during the synchronization process.
         *
         * @note
         * - The time complexity of this function depends on whether the `_requires_grad` flag is `true` and the time required for the CUDA stream write operations on `_grad` to complete. If `_requires_grad` is `false`, the function has a constant time complexity O(1). Otherwise, in the worst - case scenario with long - running write operations, it may take a significant amount of time.
         *
         * @code
         * ```cpp
         * // Assume Tensor is defined and an instance is created
         * Tensor tensor;
         * tensor.syncGrad();
         * ```
         * @endcode
         */
        void syncGrad() const;

        /**
         * @brief Synchronize both the tensor data and its gradient data.
         *
         * This function calls the `syncData` method to synchronize the tensor data and then calls the `syncGrad` method to synchronize the gradient data if gradient computation is required. It ensures that all CUDA stream write operations on the data and gradient (if applicable) are completed.
         *
         * @param None
         *
         * @return None
         *
         * Memory management for the data and gradient is assumed to be handled by the `syncData` and `syncGrad` methods respectively. There is no additional memory allocation or deallocation within this function.
         * This function does not have an explicit exception - handling mechanism. It relies on the exception - handling of the `syncData` and `syncGrad` methods to manage any errors that may occur during the synchronization process.
         *
         * @note
         * - The time complexity of this function depends on the time complexity of the `syncData` and `syncGrad` methods. In the worst - case scenario, if both operations involve long - running CUDA stream write operations, it may take a significant amount of time.
         *
         * @code
         * ```cpp
         * // Assume Tensor is defined and an instance is created
         * Tensor tensor;
         * tensor.sync();
         * ```
         * @endcode
         */
        void sync() const;

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
         * @brief Prints the tensor data to an output stream.
         *
         * @param os The output stream to which the tensor data will be written (host-to-host).
         *
         * @return The output stream after the tensor data has been written.
         *
         * This function copies the tensor data from device memory to host memory using `cudaMemcpy`.
         * It then allocates memory on the host using `malloc` to hold the copied data.
         * After printing the data to the output stream, it frees the allocated host memory using `free`.
         * The function does not throw any exceptions under normal circumstances. If `cudaMemcpy` fails,
         * the behavior depends on the `CHECK` macro, which is assumed to handle errors appropriately.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the total number of elements in the tensor.
         * - Ensure that the CUDA environment is properly initialized before calling this function.
         *
         * @code
         * ```cpp
         * Tensor tensor;
         * std::ostringstream oss;
         * tensor.print(oss);
         * std::cout << oss.str();
         * ```
         * @endcode
         */
        std::ostream& print(std::ostream& os) const;

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

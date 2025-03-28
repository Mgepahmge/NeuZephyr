#ifndef MAPPEDTENSOR_CUH
#define MAPPEDTENSOR_CUH
#include <stdexcept>
#include <vector>
#include "dl_export.cuh"
#include "Dimension.cuh"

namespace nz::data {
    /**
     * @class MappedTensor
     * @brief A class for representing multidimensional arrays in CUDA zero-copy memory, providing host-accessible container-like interfaces.
     *
     * The `MappedTensor` class offers similar functionality to the `Tensor` class but with data stored in
     * pinned zero-copy memory. This design enables direct host/device memory access patterns and container-style
     * operations at the cost of reduced computational performance compared to regular GPU memory.
     *
     * ### Type Definitions:
     * - `size_type`: Alias for `unsigned long long`, supports 64-bit indexing for large tensors.
     * - `value_type`: Alias for `float`, consistent with standard numerical computation types.
     * - `shape_type`: Alias for `std::vector<int>`, represents tensor dimensions (e.g., `{256, 256}` for an image tensor).
     * - `iterator`: Alias for `value_type*`, provides STL-style iterator access to tensor elements.
     *
     * @details
     * ### Key Differentiators from Tensor:
     * - **Zero-Copy Memory**: Utilizes CUDA pinned memory accessible by both host and device without explicit transfers
     * - **Host-Side Interoperability**: Supports STL-style iterators, range-based loops, and direct data access like `std::vector`
     * - **Container Compatibility**: Works seamlessly with standard algorithms (`std::copy`, `std::transform`, etc.)
     * - **Performance Tradeoff**: Optimized for accessibility over speed, suitable for IO-bound operations
     *
     * ### Recommended Use Cases:
     * - Frequent host-device data exchange scenarios
     * - Prototyping with direct host-side data manipulation
     * - Situations requiring container semantics with GPU data
     *
     * ### Usage Example:
     * ```cpp
     * using namespace nz::data;
     *
     * // Create mapped tensor with 3x3 shape
     * MappedTensor mtensor({3, 3});
     *
     * // Host-accessible modification via iterators
     * std::fill(mtensor.begin(), mtensor.end(), 1.0f);
     *
     * // Direct host-side data processing
     * for(auto& val : mtensor) {
     *     val = std::sqrt(val);
     * }
     *
     * // Seamless GPU computation
     * Tensor result = ReLU(mtensor);  // Works with existing Tensor operations
     * ```
     *
     * @note
     * - **Memory Characteristics**: Zero-copy memory typically offers higher allocation latency but unified access
     * - **Concurrency Considerations**: Ensure proper synchronization between host/device accesses
     * - **Performance Guidance**: Prefer Tensor for compute-intensive kernels, use MappedTensor for data pipelines
     * - **Lifecycle Management**: Pinned memory requires careful resource management - prefer RAII patterns
     *
     * @author
     * Mgepahmge(https://github.com/Mgepahmge)
     *
     * @date
     * 2024/11/29
     */
    class DL_API MappedTensor {
    public:
        using size_type = unsigned long long;
        using value_type = float;
        using shape_type = Dimension;
        using iterator = value_type*;

        friend DL_API std::ostream& operator<<(std::ostream& os, const MappedTensor& tensor);
        friend DL_API std::istream& operator>>(std::istream& is, MappedTensor& tensor);

        /// @name Constructors and Destructors
        /// @{
        /**
         * @brief Constructs a MappedTensor object.
         *
         * @param shape The shape of the tensor. This is a reference (host-to-host) to the shape_type object which defines the dimensions of the tensor.
         * @param requires_grad A boolean value indicating whether the tensor requires gradient computation.
         *
         * @return None (constructor).
         *
         * This constructor initializes the MappedTensor object with the given shape and gradient requirement. It calculates the size of the tensor based on the provided shape. Memory for the data buffer is allocated using `cudaMallocHost`. If the `requires_grad` flag is set to `true`, memory for the gradient buffer is also allocated using `cudaMallocHost`. If `requires_grad` is `false`, the gradient buffer pointer is set to `nullptr`. The memory allocated by `cudaMallocHost` should be freed by the corresponding `cudaFreeHost` call when it is no longer needed. There is no explicit exception handling in this constructor, but the `CHECK` macro is assumed to handle errors related to CUDA memory allocation. This constructor is a fundamental part of the `MappedTensor` class and is used to initialize new tensor objects.
         *
         * @note
         * - The `CHECK` macro is assumed to handle CUDA errors properly. Ensure that the CUDA environment is properly configured before using this constructor.
         *
         * @warning
         * - CUDA memory allocation may fail if there is not enough available memory on the host. Ensure that the host system has sufficient memory before creating a MappedTensor object.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * ```
         * @endcode
         */
        explicit MappedTensor(const shape_type& shape, bool requires_grad = false);

        /**
         * @brief Constructs a default MappedTensor object.
         *
         * @param None
         *
         * @return None (constructor).
         *
         * This default constructor initializes the MappedTensor object by delegating to the parameterized constructor with a shape of {0, 0} and a `requires_grad` value of `false`. After the delegation, it explicitly sets the `_data` and `_grad` pointers to `nullptr`. The memory management strategy relies on the parameterized constructor's behavior. Since the shape is {0, 0}, it's likely that no actual memory will be allocated for data and gradient in this case. There is no explicit exception - handling in this constructor, but it depends on the error - handling of the parameterized constructor (presumably through the `CHECK` macro). This constructor provides a way to create a default - initialized MappedTensor object.
         *
         * @note
         * - Ensure that the parameterized constructor is implemented correctly as this constructor depends on it.
         * - Since the `_data` and `_grad` pointers are set to `nullptr`, this object may not be suitable for direct use without re - initializing.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor tensor;
         * ```
         * @endcode
         */
        MappedTensor();

        /**
         * @brief Copy constructor for the MappedTensor class.
         *
         * @param other A constant reference (host-to-host) to another MappedTensor object from which the data and properties will be copied.
         *
         * @return None (constructor).
         *
         * This copy constructor initializes a new MappedTensor object by delegating to the parameterized constructor with the shape and gradient requirement of the `other` MappedTensor. It then copies the data from the `other` object to the newly created object using `cudaMemcpy`. If the `requires_grad` flag is set to `true`, it also copies the gradient data. The memory for the new object is already allocated by the delegated constructor. The `cudaMemcpy` operations are used to transfer data between device memory locations. There is no explicit exception handling in this constructor, but the `CHECK` macro is assumed to handle errors related to CUDA memory copy operations. This constructor is important for creating a deep copy of a MappedTensor object.
         *
         * @note
         * - The `CHECK` macro is assumed to handle CUDA errors properly. Ensure that the CUDA environment is properly configured before using this copy constructor.
         * - The `cudaMemcpy` operations may fail if there is not enough available memory or if the memory pointers are invalid.
         *
         * @warning
         * - CUDA memory copy operations may be time - consuming, especially for large tensors. Be aware of the performance implications when using this copy constructor.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor original(shape, true);
         * nz::data::MappedTensor copy(original);
         * ```
         * @endcode
         */
        MappedTensor(const MappedTensor& other);

        /**
         * @brief Move constructor for the MappedTensor class.
         *
         * @param other An rvalue reference (host-to-host) to a MappedTensor object from which resources will be moved.
         *
         * @return None (constructor).
         *
         * This move constructor transfers ownership of the resources (such as the data and gradient buffers) from the `other` MappedTensor object to the newly constructed object. It copies the shape, size, and `requires_grad` flag from the `other` object. Then, it takes over the pointers to the data and gradient buffers, leaving the `other` object in a valid but empty state. The `other` object's data and gradient pointers are set to `nullptr`, its size is set to 0, `requires_grad` is set to `false`, and the shape is set to `{0, 0}`. This operation is noexcept, meaning it does not throw exceptions. The memory management strategy is that the ownership of the previously allocated memory by the `other` object is transferred, and no new memory is allocated in this constructor. There is no need for explicit exception handling as the operation is guaranteed not to throw. This constructor is useful for efficient resource transfer during operations like returning a MappedTensor object from a function.
         *
         * @note
         * - After the move operation, the `other` object should not be used as it is left in a valid but empty state.
         * - This move constructor ensures efficient resource utilization by avoiding unnecessary memory copying.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor original(shape, true);
         * nz::data::MappedTensor moved(std::move(original));
         * ```
         * @endcode
         */
        MappedTensor(MappedTensor&& other) noexcept;

        /**
         * @brief Copy assignment operator for the MappedTensor class.
         *
         * @param other A constant reference (host-to-host) to a MappedTensor object from which data and properties will be copied.
         *
         * @return A reference to the modified MappedTensor object (host-to-host).
         *
         * This copy assignment operator first checks if the object is not being assigned to itself. If not, it releases the previously allocated memory for the `_data` and, if `_requires_grad` is true, for the `_grad` using `cudaFreeHost`. Then it copies the shape, size, and `_requires_grad` flag from the `other` object. Next, it allocates new host memory for `_data` and, if `_requires_grad` is true, for `_grad` using `cudaMallocHost`. If `_requires_grad` is false, the `_grad` pointer is set to `nullptr`. Finally, it copies the data and, if applicable, the gradient from the `other` object using `cudaMemcpy`. The memory management strategy involves deallocating existing memory before re - allocating and copying new data. There is no explicit exception handling in this operator, but the `CHECK` macro is assumed to handle errors related to CUDA memory operations. This operator is used to assign the state of one MappedTensor object to another.
         *
         * @note
         * - The `CHECK` macro is assumed to handle CUDA errors properly. Ensure that the CUDA environment is properly configured before using this operator.
         * - The CUDA memory operations may fail if there is not enough available memory or if the memory pointers are invalid.
         *
         * @warning
         * - CUDA memory operations, such as `cudaMallocHost` and `cudaMemcpy`, can be time - consuming, especially for large tensors. Be aware of the performance implications when using this operator.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor1(shape, true);
         * nz::data::MappedTensor tensor2(shape, false);
         * tensor2 = tensor1;
         * ```
         * @endcode
         */
        MappedTensor& operator=(const MappedTensor& other);

        /**
         * @brief Move assignment operator for the MappedTensor class.
         *
         * @param other An rvalue reference (host-to-host) to a MappedTensor object from which resources will be moved.
         *
         * @return A reference to the modified MappedTensor object (host-to-host).
         *
         * This move assignment operator first checks if the object is not being assigned to itself. If so, it releases the existing host memory allocated for `_data` and, if `_requires_grad` is true and `_grad` is not null, for `_grad` using `cudaFreeHost`.
         * Then, it transfers the ownership of resources from the `other` MappedTensor object to the current one. It moves the shape using `std::move`, copies the size and `_requires_grad` flag, and takes over the pointers to the data and gradient buffers from `other`.
         * After that, it sets the `other` object's data and gradient pointers to `nullptr`, its size to 0, `_requires_grad` to `false`, and the shape to `{0, 0}`. This operation is marked as `noexcept`, meaning it does not throw exceptions.
         * The memory management strategy involves freeing the current object's existing memory before taking over the memory from `other`, thus ensuring no memory leaks. There is no need for explicit exception handling as the operation is guaranteed not to throw. This operator is useful for efficiently reusing resources during assignment operations.
         *
         * @note
         * - After the move operation, the `other` object should not be used as it is left in a valid but empty state.
         * - The `CHECK` macro is assumed to handle CUDA errors properly. Ensure that the CUDA environment is properly configured before using this operator.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor1(shape, true);
         * nz::data::MappedTensor tensor2(shape, false);
         * tensor2 = std::move(tensor1);
         * ```
         * @endcode
         */
        MappedTensor& operator=(MappedTensor&& other) noexcept(false);

        /**
         * @brief Destructor for the MappedTensor class.
         *
         * This destructor is responsible for releasing the host memory allocated for the MappedTensor object.
         *
         * @details
         * - First, it checks if the `_requires_grad` flag is set to `true` and the `_grad` pointer is not `nullptr`. If so, it uses `cudaFreeHost` to free the host memory associated with the gradient data. The `CHECK` macro is assumed to handle any CUDA errors that may occur during this operation.
         * - Then, it checks if the `_data` pointer is not `nullptr`. If true, it uses `cudaFreeHost` to free the host memory associated with the tensor data, again relying on the `CHECK` macro to handle potential CUDA errors.
         *
         * The memory management strategy of this destructor ensures that any host memory allocated by the MappedTensor object is properly freed when the object goes out of scope, preventing memory leaks. There is no explicit exception handling in the destructor itself, but the `CHECK` macro is assumed to manage CUDA - related errors.
         *
         * @note
         * - The `CHECK` macro is assumed to handle CUDA errors properly. Ensure that the CUDA environment is properly configured before the destructor is called.
         * - The destructor is automatically called when the MappedTensor object goes out of scope or is explicitly deleted.
         *
         * @code
         * ```cpp
         * {
         *     nz::data::MappedTensor::shape_type shape = {2, 3};
         *     nz::data::MappedTensor tensor(shape, true);
         *     // tensor goes out of scope here, and the destructor is called automatically
         * }
         * ```
         * @endcode
         */
        ~MappedTensor() noexcept(false);
        /// @}

        /// @name Getters and Setters
        /// @{

        /**
         * @brief Returns an iterator pointing to the first element of the MappedTensor.
         *
         * @return An iterator (host-to-host) of type `MappedTensor::iterator` pointing to the first element of the tensor's data.
         *
         * This function provides a way to access the first element of the MappedTensor in a sequential manner. It simply returns the pointer `_data` as an iterator, allowing users to traverse the tensor's elements using standard iterator operations.
         *
         * The memory management strategy is to rely on the existing memory allocation of the MappedTensor object. The iterator points to the same memory location as `_data`, and no new memory is allocated or freed in this function.
         * There is no explicit exception handling in this function, as it is a simple pointer return operation and is not expected to throw exceptions under normal circumstances.
         * This function is often used in combination with other standard library algorithms and range - based for loops to iterate over the tensor's elements.
         *
         * @note
         * - Ensure that the MappedTensor object is properly initialized before calling this function, as an uninitialized object may lead to undefined behavior.
         * - The returned iterator is valid as long as the MappedTensor object exists and its underlying data is not reallocated or modified in a way that invalidates the pointer.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, false);
         * nz::data::MappedTensor::iterator it = tensor.begin();
         * ```
         * @endcode
         */
        [[nodiscard]] iterator begin() const;

        /**
         * @brief Returns an iterator pointing to the past - the - end element of the MappedTensor.
         *
         * @return An iterator (host-to-host) of type `MappedTensor::iterator` pointing to the past - the - end element of the tensor's data.
         *
         * This function is used to mark the end of the range of elements in the MappedTensor. It calculates the iterator by adding the `_size` of the tensor to the `_data` pointer. This allows for standard iteration techniques where the loop continues until the iterator reaches the `end()` iterator.
         *
         * The memory management strategy is to rely on the existing memory allocation of the MappedTensor object. The iterator points to a memory location just past the last element of the tensor's data, and no new memory is allocated or freed in this function.
         * There is no explicit exception handling in this function, as it is a simple pointer arithmetic operation and is not expected to throw exceptions under normal circumstances.
         * This function is commonly used in combination with `begin()` to iterate over all the elements of the MappedTensor using standard library algorithms or range - based for loops.
         *
         * @note
         * - Ensure that the MappedTensor object is properly initialized before calling this function, as an uninitialized object may lead to undefined behavior.
         * - The returned iterator is valid as long as the MappedTensor object exists and its underlying data is not reallocated or modified in a way that invalidates the pointer.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, false);
         * nz::data::MappedTensor::iterator it_end = tensor.end();
         * ```
         * @endcode
         */
        [[nodiscard]] iterator end() const;

        /**
         * @brief Checks whether the MappedTensor requires gradient computation.
         *
         * @return A boolean value (host-to-host) indicating whether the MappedTensor requires gradient computation. `true` means it requires gradient computation, and `false` means it does not.
         *
         * This function provides a simple way to query the gradient requirement status of the MappedTensor. It is a read - only operation that accesses the internal state of the object.
         *
         * The memory management strategy is straightforward. No new memory is allocated or freed during this function call. It simply reads the internal state of the MappedTensor object.
         * There is no exception handling mechanism in this function because it is declared `noexcept`, which means it will not throw any exceptions under normal circumstances.
         * This function can be used in various parts of the codebase to determine whether certain gradient - related operations should be performed on the MappedTensor.
         *
         * @note
         * - This function is a const member function, so it can be called on const MappedTensor objects.
         * - The `[[nodiscard]]` attribute indicates that the return value should not be ignored, as it conveys important information about the gradient requirement of the tensor.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * bool grad_required = tensor.requiresGrad();
         * ```
         * @endcode
         */
        [[nodiscard]] bool requiresGrad() const noexcept;

        /**
         * @brief Retrieves a pointer to the underlying data array of the MappedTensor.
         *
         * @return A pointer (host-to-host) of type `value_type*` that points to the first element of the MappedTensor's data array.
         *
         * This function offers direct access to the raw data stored within the MappedTensor. It is useful for operations that require low - level manipulation of the data, such as interacting with external libraries that expect a raw pointer.
         *
         * The memory management strategy is to return a pointer to the existing memory allocated for the MappedTensor. No new memory is allocated or freed during this function call. The caller should not attempt to deallocate the returned pointer, as the memory is managed by the MappedTensor object.
         * There is no exception handling mechanism in this function because it is declared `noexcept`, meaning it will not throw any exceptions under normal circumstances.
         * This function can be used in combination with other functions that operate on raw data pointers, facilitating seamless integration with other parts of the system.
         *
         * @note
         * - This function is a const member function, so it can be called on const MappedTensor objects.
         * - The `[[nodiscard]]` attribute indicates that the return value should not be ignored, as it provides access to the core data of the MappedTensor.
         * - Ensure that the MappedTensor object is valid when using the returned pointer, as an invalid object may lead to undefined behavior.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, false);
         * nz::data::MappedTensor::value_type* ptr = tensor.data();
         * ```
         * @endcode
         */
        [[nodiscard]] value_type* data() const noexcept;

        /**
         * @brief Retrieves the total number of elements in the MappedTensor.
         *
         * @return A value of type `size_type` (host-to-host) representing the total number of elements in the MappedTensor.
         *
         * This function is used to obtain the size of the MappedTensor, which is the product of the dimensions of its shape. It provides a quick way to know the quantity of elements stored in the tensor.
         *
         * The memory management strategy is straightforward. No new memory is allocated or freed during this function call. It simply returns the pre - calculated size of the tensor.
         * There is no exception handling mechanism in this function because it is declared `noexcept`, meaning it will not throw any exceptions under normal circumstances.
         * This function can be used in various scenarios, such as loop iterations over all elements of the tensor or to allocate appropriate memory when transferring data to another structure.
         *
         * @note
         * - This function is a const member function, so it can be called on const MappedTensor objects.
         * - The `[[nodiscard]]` attribute indicates that the return value should not be ignored, as it gives important information about the size of the tensor.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, false);
         * nz::data::MappedTensor::size_type tensorSize = tensor.size();
         * ```
         * @endcode
         */
        [[nodiscard]] size_type size() const noexcept;

        /**
         * @brief Retrieves the shape of the MappedTensor.
         *
         * @return A value of type `shape_type` (host-to-host) representing the shape of the MappedTensor. The shape is a container that holds the size of each dimension of the tensor.
         *
         * This function allows users to obtain the dimensional information of the MappedTensor. The shape provides crucial details about how the elements are organized in the tensor, which is essential for many tensor operations.
         *
         * The memory management strategy involves returning a copy of the internal shape representation of the MappedTensor. The caller is responsible for managing the memory of the returned `shape_type` object, but the original shape data within the MappedTensor remains intact. No new memory is allocated for the tensor's internal shape data during this call.
         * There is no exception handling mechanism in this function because it is declared `noexcept`, meaning it will not throw any exceptions under normal circumstances.
         * This function can be used in combination with other functions that require knowledge of the tensor's shape, such as reshaping operations or accessing specific elements based on multi - dimensional indices.
         *
         * @note
         * - This function is a const member function, so it can be called on const MappedTensor objects.
         * - The `[[nodiscard]]` attribute indicates that the return value should not be ignored, as the shape information is vital for working with the tensor.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type inputShape = {2, 3};
         * nz::data::MappedTensor tensor(inputShape, false);
         * nz::data::MappedTensor::shape_type tensorShape = tensor.shape();
         * ```
         * @endcode
         */
        [[nodiscard]] shape_type shape() const noexcept;

        /**
         * @brief Sets the gradient requirement flag for the MappedTensor and manages the associated gradient memory accordingly.
         *
         * @param requires_grad A boolean value (host-to-host) indicating whether the tensor should require gradient computation.
         *
         * @return void
         *
         * This function is responsible for updating the `_requires_grad` flag of the MappedTensor. If the flag is being changed from `true` to `false`, it frees the memory allocated for the gradient (`_grad`). Conversely, if the flag is being changed from `false` to `true`, it allocates memory for the gradient using `cudaMallocHost`.
         *
         * The memory management strategy is as follows: When `requires_grad` is set to `false` and the tensor previously required gradients, the gradient memory is freed using `cudaFreeHost`. When `requires_grad` is set to `true` and the tensor did not previously require gradients, new memory is allocated using `cudaMallocHost`. The size of the allocated memory is based on the total number of elements (`_size`) in the tensor multiplied by the size of each element (`sizeof(size_type)`).
         * The exception handling mechanism relies on the `CHECK` macro. If the `cudaFreeHost` or `cudaMallocHost` operations fail, the `CHECK` macro is expected to handle the error appropriately, potentially throwing an exception or terminating the program.
         * This function is closely related to the gradient computation mechanism of the MappedTensor. It ensures that the memory for gradients is allocated and freed as needed, which is crucial for efficient gradient calculation and memory management in a CUDA - enabled environment.
         *
         * @throws An exception might be thrown by the `CHECK` macro if the `cudaFreeHost` or `cudaMallocHost` operations fail.
         *
         * @note
         * - Ensure that the CUDA runtime environment is properly initialized before calling this function, as it uses CUDA memory management functions.
         * - The `CHECK` macro is assumed to handle CUDA errors correctly. Any issues with the CUDA operations will be reported through this macro.
         *
         * @warning
         * - Incorrect usage of this function can lead to memory leaks or segmentation faults. For example, if the CUDA environment is not set up correctly, the memory allocation or deallocation operations may fail unexpectedly.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, false);
         * tensor.setRequiresGrad(true);
         * ```
         * @endcode
         */
        void setRequiresGrad(bool requires_grad);

        /**
         * @brief Sets a new shape for the MappedTensor and adjusts its data and gradient memory accordingly.
         *
         * @param shape A reference to a `shape_type` object (host-to-host) that represents the new shape of the MappedTensor.
         *
         * @return void
         *
         * This function is used to change the shape of the MappedTensor. It allocates new memory for the tensor's data based on the new shape, initializes the new memory to zero, copies the data from the old memory to the new memory, and then frees the old memory. If the tensor requires gradient computation (`_requires_grad` is `true`), it performs similar operations for the gradient memory.
         *
         * The memory management strategy involves multiple steps. First, new memory is allocated using `cudaMallocHost` for the data and gradient (if required). Then, the new memory is initialized to zero using `cudaMemset`. Next, the data is copied from the old memory to the new memory using `cudaMemcpy` with the `cudaMemcpyDeviceToDevice` flag. Finally, the old memory is freed using `cudaFreeHost`.
         * The exception handling mechanism relies on the `CHECK` macro. If any of the CUDA operations (`cudaMallocHost`, `cudaMemset`, `cudaMemcpy`, `cudaFreeHost`) fail, the `CHECK` macro is expected to handle the error, potentially throwing an exception or terminating the program.
         * This function is closely related to the data storage and gradient management components of the MappedTensor. It ensures that the tensor's data and gradient are properly adjusted when the shape changes, which is crucial for maintaining data consistency during tensor operations.
         *
         * @throws An exception might be thrown by the `CHECK` macro if any of the CUDA operations fail.
         *
         * @note
         * - Ensure that the CUDA runtime environment is properly initialized before calling this function, as it uses CUDA memory management and data transfer functions.
         * - The `CHECK` macro is assumed to handle CUDA errors correctly. Any issues with the CUDA operations will be reported through this macro.
         * - The function assumes that the `shape_type` object has at least two elements, as it accesses `shape[0]` and `shape[1]` to calculate the new size.
         *
         * @warning
         * - Incorrect usage of this function can lead to memory leaks or segmentation faults. For example, if the CUDA environment is not set up correctly, the memory allocation, initialization, or deallocation operations may fail unexpectedly.
         * - If the new shape has a smaller size than the old shape, data beyond the new size will be discarded. If the new shape has a larger size, the additional memory will be initialized to zero.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type oldShape = {2, 3};
         * nz::data::MappedTensor tensor(oldShape, false);
         * nz::data::MappedTensor::shape_type newShape = {3, 2};
         * tensor.setShape(newShape);
         * ```
         * @endcode
         */
        void setShape(const shape_type& shape);

        /**
         * @brief Inject data into either the tensor's main data or its gradient.
         *
         * @param data A pointer to an array of `value_type` (host-to-device) that contains the data to be injected.
         * @param size A `size_type` value (host-to-device) representing the number of elements in the `data` array.
         * @param isGrad A boolean value (host-to-device) indicating whether the data should be injected into the gradient (`true`) or the main tensor data (`false`).
         *
         * @return void
         *
         * This function is designed to inject external data into the MappedTensor. Depending on the `isGrad` flag, it will copy the provided data either into the tensor's main data or its gradient.
         *
         * Memory management: The caller is responsible for allocating and deallocating the memory of the `data` array on the host side. The function uses `cudaMemcpy` to transfer the data from the host to the device memory of the tensor or its gradient. It only copies the minimum of `size` and `_size` elements to prevent out - of - bounds access.
         * Exception handling: If `isGrad` is `true` and the tensor does not require gradients (`_requires_grad` is `false`), a `std::invalid_argument` exception is thrown. Additionally, if the `cudaMemcpy` operation fails, the `CHECK` macro is expected to handle the error, potentially throwing an exception or terminating the program.
         * Relationship with other components: This function is closely related to the data storage and gradient management components of the MappedTensor. It provides a way to update the tensor's data or gradient values from an external source.
         *
         * @throws std::invalid_argument If `isGrad` is `true` and the tensor does not require gradients.
         * @throws An exception might be thrown by the `CHECK` macro if the `cudaMemcpy` operation fails.
         *
         * @note
         * - Ensure that the CUDA runtime environment is properly initialized before calling this function, as it uses `cudaMemcpy` for data transfer.
         * - The `CHECK` macro is assumed to handle CUDA errors correctly. Any issues with the CUDA operations will be reported through this macro.
         * - The function only copies the minimum of `size` and `_size` elements to avoid out - of - bounds access.
         *
         * @warning
         * - Incorrect usage of this function can lead to data corruption or CUDA errors. For example, if the CUDA environment is not set up correctly, the `cudaMemcpy` operation may fail unexpectedly.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * nz::data::MappedTensor::value_type data[] = {1, 2, 3, 4, 5, 6};
         * tensor.dataInject(data, 6, false);
         * ```
         * @endcode
         */
        void dataInject(float* data, size_type size, bool isGrad = false) const;

        /**
         * @brief Inject data from an iterator range into either the tensor's data or its gradient.
         *
         * @tparam Iterator The type of the iterator used to access the data source.
         * @param begin An iterator (host-to-host) pointing to the start of the data range to be injected.
         * @param end An iterator (host-to-host) pointing past the end of the data range to be injected.
         * @param isGrad A boolean value (host-to-host) indicating whether to inject the data into the gradient (`true`) or the tensor data (`false`). Defaults to `false`.
         *
         * @return void
         *
         * This function is designed to transfer data from an iterator range into the MappedTensor. It iterates through the given range and assigns the values to either the tensor's main data or its gradient, based on the `isGrad` flag.
         *
         * Memory management: The caller is responsible for the memory occupied by the data source pointed to by the iterators. The function only reads values from the iterators and copies them into the tensor's internal memory. It ensures that at most the minimum of the range size and the tensor's size (`_size`) is copied to prevent out-of-bounds access.
         * Exception handling: If `isGrad` is `true` and the tensor does not require gradients (`_requires_grad` is `false`), a `std::invalid_argument` exception is thrown. Additionally, the function assumes that the iterators are valid and well-behaved. If operations on the iterators (such as `std::distance`, dereferencing, or incrementing) throw exceptions, those exceptions will be propagated.
         * Relationship with other components: This function interacts closely with the data storage and gradient management components of the MappedTensor. It provides a flexible way to update the tensor's data or gradient values using different data sources accessible via iterators.
         *
         * @throws std::invalid_argument If `isGrad` is `true` and the tensor does not require gradients.
         *
         * @note
         * - Ensure that the iterators `begin` and `end` form a valid range. Using invalid iterators may lead to undefined behavior.
         * - The function has a time complexity of O(min(n, _size)), where n is the number of elements in the iterator range, as it iterates over the range to copy the data.
         * - The data from the iterators is cast to `value_type` before being assigned to the tensor or its gradient.
         *
         * @code
         * ```cpp
         * #include <vector>
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * std::vector<nz::data::MappedTensor::value_type> data = {1, 2, 3, 4, 5, 6};
         * tensor.dataInject(data.begin(), data.end(), false);
         * ```
         * @endcode
         */
        template <typename Iterator>
        void dataInject(Iterator begin, Iterator end, const bool isGrad = false) const {
            if (isGrad && !_requires_grad) {
                throw std::invalid_argument("Tensor does not require gradients");
            }
            auto size = std::distance(begin, end);
            auto it = begin;
            for (size_type i = 0; i < (size < _size ? size : _size) && it != end; ++i, ++it) {
                if (isGrad) {
                    _grad[i] = static_cast<value_type>(*it);
                }
                else {
                    _data[i] = static_cast<value_type>(*it);
                }
            }
        }

        /**
         * @brief Inject data from a std::initializer_list into either the tensor's data or its gradient.
         *
         * @param data A std::initializer_list<value_type> (host-to-host) containing the data to be injected.
         * @param isGrad A boolean value (host-to-host) indicating whether to inject the data into the gradient (`true`) or the tensor data (`false`).
         *
         * @return void
         *
         * This function transfers data from a std::initializer_list to the MappedTensor. Based on the `isGrad` flag, it assigns the values from the initializer list to either the tensor's main data or its gradient.
         *
         * Memory management: The caller is responsible for the memory of the std::initializer_list. The function only reads values from the list and copies them into the tensor's internal memory. It ensures that at most the minimum of the list size and the tensor's size (`_size`) is copied to prevent out-of-bounds access.
         * Exception handling: If `isGrad` is `true` and the tensor does not require gradients (`_requires_grad` is `false`), a `std::invalid_argument` exception is thrown.
         * Relationship with other components: This function interacts with the data storage and gradient management components of the MappedTensor. It provides a convenient way to initialize the tensor's data or gradient using an initializer list.
         *
         * @throws std::invalid_argument If `isGrad` is `true` and the tensor does not require gradients.
         *
         * @note
         * - The function has a time complexity of O(min(n, _size)), where n is the number of elements in the std::initializer_list, as it iterates over the list to copy the data.
         * - Ensure that the std::initializer_list contains elements of the correct type (`value_type`).
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * tensor.dataInject({1, 2, 3, 4, 5, 6}, false);
         * ```
         * @endcode
         */
        void dataInject(const std::initializer_list<value_type>& data, const bool isGrad = false) const;

        /**
         * @brief Overload the [] operator to access an element of the MappedTensor by index.
         *
         * @param index The index of the element to access (host-to-host). It should be a non-negative integer.
         *
         * @return A reference to the value at the specified index in the tensor's data.
         *
         * This function allows users to access individual elements of the MappedTensor using the [] operator. It first checks if the given index is within the valid range of the tensor's size. If the index is out of range, it throws a `std::out_of_range` exception. Otherwise, it returns a reference to the corresponding element in the internal data array `_data`.
         *
         * Memory management: The function does not allocate or deallocate any memory. It only accesses the existing internal data array `_data`.
         * Exception handling: If the provided index is greater than or equal to the size of the tensor (`_size`), a `std::out_of_range` exception is thrown.
         * Relationship with other components: This function is related to the data access component of the MappedTensor. It provides a convenient way for users to access individual elements of the tensor.
         *
         * @throws std::out_of_range If the provided index is out of the valid range (i.e., `index >= _size`).
         *
         * @note
         * - The time complexity of this function is O(1) because it directly accesses the element in the internal data array using the given index.
         * - Ensure that the index is within the valid range to avoid exceptions.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, false);
         * tensor.dataInject({1, 2, 3, 4, 5, 6}, false);
         * nz::data::MappedTensor::value_type value = tensor[2];
         * std::cout << value << std::endl;
         * ```
         * @endcode
         */
        auto operator[](size_type index) const -> value_type&;
        /// @}

        /// @name Printer
        /// @{

        /**
         * @brief Print the tensor data in a matrix-like format to an output stream.
         *
         * @param os An output stream (host-to-host) where the tensor data will be printed.
         *
         * @return A reference to the output stream `os` after printing the tensor data.
         *
         * This function is used to display the tensor data in a matrix-like structure. It iterates over the rows of the tensor and prints each row as a sequence of values separated by a space, enclosed in square brackets.
         *
         * Memory management: The function does not allocate or deallocate any memory. It only reads the tensor's internal data (`_data`) for printing.
         * Exception handling: The function assumes that the `_shape` and `_data` members of the tensor are properly initialized. If there are issues with these members (e.g., invalid shape dimensions), the behavior may be undefined. The operations on the output stream (`os`) are assumed to be well - behaved, and any exceptions thrown by the stream operations will be propagated.
         * Relationship with other components: This function is mainly related to the data presentation component of the MappedTensor. It provides a user - friendly way to view the tensor's data.
         *
         * @note
         * - The function has a time complexity of O(m * n), where m is the number of rows (`_shape[0]`) and n is the number of columns (`_shape[1]`) of the tensor, as it iterates over all elements in the tensor.
         * - Ensure that the output stream `os` is in a valid state before calling this function.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, false);
         * tensor.dataInject({1, 2, 3, 4, 5, 6}, false);
         * tensor.print(std::cout);
         * ```
         * @endcode
         */
        std::ostream& print(std::ostream& os) const;

        /**
         * @brief Print the gradient of the tensor in a matrix-like format to an output stream.
         *
         * @param os An output stream (host-to-host) where the tensor gradient will be printed.
         *
         * @return A reference to the output stream `os` after printing the tensor gradient.
         *
         * This function is designed to display the gradient of the MappedTensor in a matrix-like structure. It iterates over the rows of the gradient data and prints each row as a sequence of values separated by a space, enclosed in square brackets.
         *
         * Memory management: The function does not allocate or deallocate any memory. It only reads the tensor's internal gradient data (`_grad`) for printing.
         * Exception handling: If the tensor does not require gradients (`_requires_grad` is `false`), a `std::invalid_argument` exception is thrown. The operations on the output stream (`os`) are assumed to be well - behaved, and any exceptions thrown by the stream operations will be propagated.
         * Relationship with other components: This function is related to the gradient management and data presentation components of the MappedTensor. It provides a way to view the gradient values of the tensor.
         *
         * @throws std::invalid_argument If the tensor does not require gradients.
         *
         * @note
         * - The function has a time complexity of O(m * n), where m is the number of rows (`_shape[0]`) and n is the number of columns (`_shape[1]`) of the tensor's gradient, as it iterates over all elements in the gradient.
         * - Ensure that the output stream `os` is in a valid state before calling this function.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * tensor.dataInject({1, 2, 3, 4, 5, 6}, true);
         * tensor.printGrad(std::cout);
         * ```
         * @endcode
         */
        std::ostream& printGrad(std::ostream& os) const;
        /// @}

        /// @name Modifiers
        /// @{

        /**
         * @brief Clear the data stored in the MappedTensor by setting all elements to zero.
         *
         * @param None
         *
         * @return None
         *
         * This function uses CUDA's `cudaMemset` to set all elements of the `_data` array to zero. It is designed to quickly reset the tensor's data.
         *
         * Memory management: The function does not allocate or deallocate memory. It simply modifies the existing data in the `_data` array.
         * Exception handling: The `CHECK` macro is used to handle potential CUDA errors. If `cudaMemset` fails, the `CHECK` macro will handle the error according to its implementation, which may include logging and terminating the program.
         * Relationship with other components: This function is related to the data management component of the MappedTensor. It provides a way to reset the tensor's data to a known state.
         *
         * @throws None explicitly, but the `CHECK` macro may handle and report CUDA errors.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), as `cudaMemset` needs to set each element to zero.
         * - Ensure that the CUDA environment is properly initialized before calling this function.
         *
         * @warning
         * - If the CUDA environment is not set up correctly, `cudaMemset` may fail, and the `CHECK` macro will handle the error, which may lead to program termination.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, false);
         * tensor.dataInject({1, 2, 3, 4, 5, 6}, false);
         * tensor.clear();
         * ```
         * @endcode
         */
        void clear() const;

        /**
         * @brief Clear the gradient data of the MappedTensor if it requires gradients.
         *
         * @param None
         *
         * @return None
         *
         * This function is used to reset the gradient data of a MappedTensor to zero. It first checks if the tensor requires gradients. If it does, it uses CUDA's `cudaMemset` to set all elements of the `_grad` array to zero. Otherwise, it throws a `std::runtime_error`.
         *
         * Memory management: The function does not allocate or deallocate memory. It only modifies the existing `_grad` array.
         * Exception handling: If the tensor does not require gradients, a `std::runtime_error` is thrown. If `cudaMemset` fails, the `CHECK` macro will handle the CUDA error according to its implementation.
         * Relationship with other components: This function is related to the gradient management component of the MappedTensor. It provides a way to reset the gradient data for tensors that support gradient computation.
         *
         * @throws std::runtime_error If the tensor does not require gradients.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), as `cudaMemset` needs to set each element of the gradient array to zero.
         * - Ensure that the CUDA environment is properly initialized before calling this function if the tensor requires gradients.
         *
         * @warning
         * - Attempting to clear gradients for a tensor that does not require gradients will result in a runtime error.
         * - If the CUDA environment is not set up correctly and the tensor requires gradients, `cudaMemset` may fail, and the `CHECK` macro will handle the error, which may lead to program termination.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * try {
         *     tensor.clearGrad();
         * } catch (const std::runtime_error& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        void clearGrad() const;

        /**
         * @brief Reshape the MappedTensor to a new shape.
         *
         * @param shape The new shape of the tensor (host-to-device).
         *
         * @return None
         *
         * This function reshapes the MappedTensor to the specified new shape. It allocates a new memory block for the data and, if the tensor requires gradients, for the gradients as well. It then copies the existing data and gradients (up to the minimum of the old and new sizes) to the new memory blocks and frees the old memory. Finally, it updates the tensor's shape and size information.
         *
         * Memory management: The function allocates new host memory for the data and gradients using `cudaMallocHost` and frees the old host memory using `cudaFreeHost`.
         * Exception handling: The `CHECK` macro is used to handle potential CUDA errors during memory allocation, memory setting, and memory copying operations. If any of these operations fail, the `CHECK` macro will handle the error according to its implementation, which may include logging and terminating the program.
         * Relationship with other components: This function is related to the data and gradient management components of the MappedTensor. It provides a way to change the shape of the tensor and adjust its internal memory accordingly.
         *
         * @throws None explicitly, but the `CHECK` macro may handle and report CUDA errors.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the minimum of the old and new sizes of the tensor, as the data and gradients are copied element by element.
         * - Ensure that the CUDA environment is properly initialized before calling this function.
         * - The new shape should be compatible with the intended use of the tensor.
         *
         * @warning
         * - If the CUDA environment is not set up correctly, memory allocation, setting, or copying operations may fail, and the `CHECK` macro will handle the error, which may lead to program termination.
         * - Changing the shape of a tensor may affect the interpretation of its data in subsequent operations.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type oldShape = {2, 3};
         * nz::data::MappedTensor::shape_type newShape = {3, 2};
         * nz::data::MappedTensor tensor(oldShape, false);
         * tensor.reshape(newShape);
         * ```
         * @endcode
         */
        void reshape(const shape_type& shape);

        /**
         * @brief Randomize the data or gradients of the MappedTensor using a given seed.
         *
         * @param seed The seed value used to initialize the random number generator (host-to-device). If 0, the current system time will be used as the seed.
         * @param isGrad A boolean flag indicating whether to randomize the gradients or the data. If true, gradients are randomized; otherwise, data is randomized (host-to-device).
         *
         * @return None
         *
         * This function provides the ability to randomize either the data or the gradients of the MappedTensor. It first checks if gradient randomization is valid for the tensor. If the `seed` is 0, it uses the current system time as the seed. Then, it initializes a CURAND pseudo - random number generator, sets the seed, and fills the appropriate memory (data or gradients) with uniformly distributed random numbers in the range [0, 1).
         *
         * Memory management: The function does not allocate or deallocate the tensor's data or gradient memory. It only modifies the existing memory in - place.
         * Exception handling: If the tensor does not require gradients and `isGrad` is true, a `std::invalid_argument` is thrown. If any of the CURAND operations (creating the generator, setting the seed, or generating random numbers) fail, a `std::runtime_error` is thrown.
         * Relationship with other components: This function is related to the data and gradient initialization components of the MappedTensor. It offers a way to initialize the tensor's data or gradients with random values.
         *
         * @throws std::invalid_argument If gradient randomization is attempted on a tensor that does not require gradients.
         * @throws std::runtime_error If CURAND fails to create the generator, set the seed, or generate random numbers.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), as it needs to generate a random number for each element.
         * - Ensure that the CUDA and CURAND libraries are properly initialized and configured before calling this function.
         *
         * @warning
         * - If the CURAND library is not installed or configured correctly, this function will throw an exception.
         * - Reusing the same seed will result in the same sequence of random numbers being generated.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * try {
         *     tensor.randomize(42, false);
         *     tensor.randomize(0, true);
         * } catch (const std::exception& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        void randomize(size_type seed = 0, bool isGrad = false) const;

        /**
         * @brief Fill the data or gradients of the MappedTensor with a given value.
         *
         * @param value The value used to fill the tensor's data or gradients (host-to-device).
         * @param isGrad A boolean flag indicating whether to fill the gradients or the data. If true, gradients are filled; otherwise, data is filled (host-to-device).
         *
         * @return None
         *
         * This function fills either the data or the gradients of the MappedTensor with the specified `value`. It determines the appropriate CUDA grid and block dimensions based on the size of the tensor, and then invokes the `krnl::Fill` kernel to perform the filling operation.
         *
         * Memory management: The function does not allocate or deallocate the tensor's data or gradient memory. It only modifies the existing memory in-place.
         * Exception handling: It is assumed that the `krnl::Fill` kernel handles its own errors and may throw exceptions in case of issues. If an exception occurs in the kernel, it will propagate up.
         * Relationship with other components: This function depends on the `krnl::Fill` kernel to perform the actual filling operation. It provides a high - level interface for initializing the tensor's data or gradients.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), as it needs to set each element to the given value.
         * - Ensure that the CUDA environment is properly configured and the `krnl::Fill` kernel is correctly implemented before calling this function.
         *
         * @warning
         * - If the CUDA device is not properly initialized or the `krnl::Fill` kernel has implementation issues, this function may fail.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * tensor.fill(1.0f, false);
         * tensor.fill(0.0f, true);
         * ```
         * @endcode
         */
        void fill(value_type value, bool isGrad = false) const;

        void fillMatrix(value_type value, size_type batch, size_type channels, bool isGrad = false);

        /**
         * @brief Transpose the MappedTensor and its gradients (if required).
         *
         * @param None
         *
         * @return None
         *
         * This function transposes the data of the MappedTensor. If the tensor requires gradients, it also transposes the gradients. It first calculates the CUDA grid and block dimensions based on the tensor's shape and a predefined tile size. Then, it allocates host memory for a temporary buffer, invokes the `krnl::Transpose` kernel to perform the transpose operation, and synchronizes the device. After that, it frees the original data memory and assigns the temporary buffer as the new data. If gradients are required, the same process is repeated for the gradients. Finally, it swaps the shape dimensions of the tensor.
         *
         * Memory management: The function allocates host memory for temporary buffers using `cudaMallocHost` and frees the original data and gradient memory using `cudaFreeHost`.
         * Exception handling: The `CHECK` macro is used to handle CUDA errors. If a CUDA operation fails, the `CHECK` macro will throw an appropriate exception.
         * Relationship with other components: This function depends on the `krnl::Transpose` kernel to perform the actual transpose operation. It also interacts with the CUDA memory management functions.
         *
         * @throws [Exception type thrown by CHECK macro] [Thrown when a CUDA operation fails]
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), as it needs to process each element during the transpose operation.
         * - Ensure that the CUDA environment is properly configured and the `krnl::Transpose` kernel is correctly implemented before calling this function.
         * - The `TILE_SIZE` must be properly defined for the CUDA kernel to work correctly.
         *
         * @warning
         * - If the CUDA device runs out of memory during the memory allocation, the function will fail.
         * - Incorrect implementation of the `krnl::Transpose` kernel may lead to incorrect transpose results.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * tensor.transpose();
         * ```
         * @endcode
         */
        void transpose();
        /// @}

        /// @name Math
        /// @{

        /**
         * @brief Perform element-wise addition between two MappedTensors.
         *
         * @param other The MappedTensor to be added to the current MappedTensor (host-to-host).
         *
         * @return A new MappedTensor containing the result of the element-wise addition.
         *
         * This function performs an element-wise addition between the current MappedTensor and another MappedTensor. It first checks if the shapes of the two tensors are equal; if not, it throws a `std::invalid_argument` exception. Then, it creates a new MappedTensor with the same shape and the appropriate gradient requirement based on the two input tensors. After that, it calculates the CUDA grid and block dimensions according to the size of the tensors and invokes the `krnl::MatrixAdd` kernel to perform the actual addition operation. Finally, it synchronizes the CUDA device and returns the resulting MappedTensor.
         *
         * Memory management: A new MappedTensor is created to store the result, and its memory is managed automatically by the MappedTensor class. The input tensors' memory remains unchanged.
         * Exception handling: If the shapes of the two input tensors are not equal, a `std::invalid_argument` exception is thrown. The `CHECK` macro is used to handle CUDA errors, and if a CUDA operation fails, it will throw an appropriate exception.
         * Relationship with other components: This function depends on the `krnl::MatrixAdd` kernel to perform the element-wise addition and the `CHECK` macro to handle CUDA errors.
         *
         * @throws std::invalid_argument If the shapes of the two MappedTensors are not equal.
         * @throws [Exception type thrown by CHECK macro] If a CUDA operation fails.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensors (`_size`), as it needs to perform an addition operation for each element.
         * - Ensure that the CUDA environment is properly configured and the `krnl::MatrixAdd` kernel is correctly implemented before calling this function.
         *
         * @warning
         * - If the CUDA device runs out of memory during the operation, the function may fail.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor1(shape, true);
         * nz::data::MappedTensor tensor2(shape, false);
         * nz::data::MappedTensor result = tensor1 + tensor2;
         * ```
         * @endcode
         */
        MappedTensor operator+(const MappedTensor& other) const;

        /**
         * @brief Perform element-wise subtraction between two MappedTensors.
         *
         * @param other The MappedTensor to be subtracted from the current MappedTensor (host-to-host).
         *
         * @return A new MappedTensor containing the result of the element-wise subtraction.
         *
         * This function conducts an element-wise subtraction between the current MappedTensor and another MappedTensor. It first verifies that the shapes of the two tensors are equal; if not, it throws a `std::invalid_argument` exception. Then, it constructs a new MappedTensor with the same shape and an appropriate gradient requirement based on the input tensors. Subsequently, it calculates the CUDA grid and block dimensions according to the tensor size and invokes the `krnl::MatrixSub` kernel to carry out the subtraction operation. Finally, it synchronizes the CUDA device and returns the resulting MappedTensor.
         *
         * Memory management: A new MappedTensor is created to store the result, and its memory is managed by the MappedTensor class. The memory of the input tensors remains untouched.
         * Exception handling: Throws a `std::invalid_argument` if the shapes of the two MappedTensors do not match. The `CHECK` macro is used to handle CUDA errors; if a CUDA operation fails, it will throw an appropriate exception.
         * Relationship with other components: Depends on the `krnl::MatrixSub` kernel for the subtraction operation and the `CHECK` macro for CUDA error handling.
         *
         * @throws std::invalid_argument If the shapes of the two MappedTensors are not equal.
         * @throws [Exception type thrown by CHECK macro] If a CUDA operation fails.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensors (`_size`), as it needs to perform a subtraction for each element.
         * - Ensure that the CUDA environment is properly configured and the `krnl::MatrixSub` kernel is correctly implemented before using this function.
         *
         * @warning
         * - If the CUDA device runs out of memory during the operation, the function may fail.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor1(shape, true);
         * nz::data::MappedTensor tensor2(shape, false);
         * nz::data::MappedTensor result = tensor1 - tensor2;
         * ```
         * @endcode
         */
        MappedTensor operator-(const MappedTensor& other) const;

        /**
         * @brief Perform matrix multiplication between two MappedTensors.
         *
         * @param other The MappedTensor to be multiplied with the current MappedTensor (host-to-host).
         *
         * @return A new MappedTensor containing the result of the matrix multiplication.
         *
         * This function performs matrix multiplication between the current MappedTensor and another MappedTensor. It first checks if the number of columns in the current tensor is equal to the number of rows in the other tensor. If not, it throws a `std::invalid_argument` exception. Then, it creates a new MappedTensor with the appropriate shape for the result and sets the gradient requirement based on the current tensor. Next, it calculates the CUDA grid and block dimensions according to the shape of the result tensor and invokes the `krnl::GeneralMatrixMul` kernel to perform the actual matrix multiplication. Finally, it synchronizes the CUDA device and returns the resulting MappedTensor.
         *
         * Memory management: A new MappedTensor is created to store the result, and its memory is managed automatically by the MappedTensor class. The input tensors' memory remains unchanged.
         * Exception handling: If the matrix shapes do not match for multiplication, a `std::invalid_argument` exception is thrown. The `CHECK` macro is used to handle CUDA errors, and if a CUDA operation fails, it will throw an appropriate exception.
         * Relationship with other components: This function depends on the `krnl::GeneralMatrixMul` kernel to perform the matrix multiplication and the `CHECK` macro to handle CUDA errors.
         *
         * @throws std::invalid_argument If the number of columns in the current MappedTensor is not equal to the number of rows in the other MappedTensor.
         * @throws [Exception type thrown by CHECK macro] If a CUDA operation fails.
         *
         * @note
         * - The time complexity of this function is O(m * n * k), where m is the number of rows in the current tensor, n is the number of columns in the other tensor, and k is the number of columns in the current tensor (which is equal to the number of rows in the other tensor).
         *
         * @warning
         * - If the CUDA device runs out of memory during the operation, the function may fail.
         * - Incorrect implementation of the `krnl::GeneralMatrixMul` kernel may lead to incorrect multiplication results.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape1 = {2, 3};
         * nz::data::MappedTensor::shape_type shape2 = {3, 4};
         * nz::data::MappedTensor tensor1(shape1, true);
         * nz::data::MappedTensor tensor2(shape2, false);
         * nz::data::MappedTensor result = tensor1 * tensor2;
         * ```
         * @endcode
         */
        MappedTensor operator*(const MappedTensor& other) const;

        /**
         * @brief Perform element-wise negation on the MappedTensor.
         *
         * @return A new MappedTensor containing the element-wise negation of the current MappedTensor.
         *
         * This function performs an element-wise negation operation on the current MappedTensor. It first calculates the CUDA grid and block dimensions based on the size of the tensor. Then, it creates a new MappedTensor with the same shape and gradient requirement as the current one. After that, it invokes the `krnl::Negation` kernel to perform the negation operation on each element of the tensor. Finally, it synchronizes the CUDA device and returns the resulting MappedTensor.
         *
         * Memory management: A new MappedTensor is created to store the result, and its memory is managed by the MappedTensor class. The memory of the current MappedTensor remains unchanged.
         * Exception handling: The `CHECK` macro is used to handle CUDA errors. If a CUDA operation fails, it will throw an appropriate exception.
         * Relationship with other components: This function depends on the `krnl::Negation` kernel for the negation operation and the `CHECK` macro for CUDA error handling.
         *
         * @throws [Exception type thrown by CHECK macro] If a CUDA operation fails.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), as it needs to perform a negation for each element.
         * - Ensure that the CUDA environment is properly configured and the `krnl::Negation` kernel is correctly implemented before using this function.
         *
         * @warning
         * - If the CUDA device runs out of memory during the operation, the function may fail.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * nz::data::MappedTensor negatedTensor = -tensor;
         * ```
         * @endcode
         */
        MappedTensor operator-() const;

        /**
         * @brief Perform element-wise division between two MappedTensors.
         *
         * @param other The MappedTensor to divide the current MappedTensor by (host-to-host).
         *
         * @return A new MappedTensor containing the result of the element-wise division.
         *
         * This function performs element-wise division between the current MappedTensor and another MappedTensor. It first checks if the shapes of the two tensors are equal. If not, it throws a `std::invalid_argument` exception. Then, it calculates the CUDA grid and block dimensions based on the size of the tensors. A new MappedTensor with the same shape and gradient requirement as the current tensor is created. The `krnl::ElementwiseDivide` kernel is invoked to perform the division operation on each corresponding element of the two tensors. Finally, the CUDA device is synchronized, and the resulting MappedTensor is returned.
         *
         * Memory management: A new MappedTensor is created to store the result, and its memory is managed by the MappedTensor class. The memory of the input tensors remains unchanged.
         * Exception handling: If the shapes of the two tensors are not equal, a `std::invalid_argument` exception is thrown. The `CHECK` macro is used to handle CUDA errors, and if a CUDA operation fails, an appropriate exception is thrown.
         * Relationship with other components: This function depends on the `krnl::ElementwiseDivide` kernel for the division operation and the `CHECK` macro for CUDA error handling.
         *
         * @throws std::invalid_argument If the shapes of the two MappedTensors are not equal.
         * @throws [Exception type thrown by CHECK macro] If a CUDA operation fails.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensors (`_size`), as it needs to perform a division for each pair of corresponding elements.
         *
         * @warning
         * - If the CUDA device runs out of memory during the operation, the function may fail.
         * - Incorrect implementation of the `krnl::ElementwiseDivide` kernel may lead to incorrect division results.
         * - Division by zero in the `krnl::ElementwiseDivide` kernel may lead to undefined behavior.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor1(shape, true);
         * nz::data::MappedTensor tensor2(shape, false);
         * nz::data::MappedTensor result = tensor1 / tensor2;
         * ```
         * @endcode
         */
        MappedTensor operator/(const MappedTensor& other) const;

        /**
         * @brief Compute the reciprocal of each element in the MappedTensor.
         *
         * This function computes the reciprocal (1/x) of each element in the MappedTensor. It first calculates the CUDA grid and block dimensions based on the size of the tensor. Then, it allocates host memory for a temporary buffer to store the reciprocal values. The `krnl::Recip` kernel is invoked to compute the reciprocals of the elements in the tensor and store the results in the temporary buffer. After the kernel execution, the CUDA device is synchronized. Finally, the original device memory of the tensor is freed, and the pointer is updated to point to the temporary buffer.
         *
         * Memory management: Host memory is allocated for the temporary buffer using `cudaMallocHost`. The original device memory of the tensor is freed using `cudaFreeHost`. The ownership of the data is transferred to the `_data` member of the MappedTensor.
         * Exception handling: The `CHECK` macro is used to handle CUDA errors. If any CUDA operation fails, an appropriate exception will be thrown.
         * Relationship with other components: This function depends on the `krnl::Recip` kernel for computing the reciprocals and the `CHECK` macro for CUDA error handling.
         *
         * @throws [Exception type thrown by CHECK macro] If a CUDA operation fails.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the tensor (`_size`), as it needs to compute the reciprocal for each element.
         * - Ensure that the CUDA environment is properly configured and the `krnl::Recip` kernel is correctly implemented before using this function.
         *
         * @warning
         * - If the CUDA device runs out of memory during the operation, the function may fail.
         * - Division by zero in the `krnl::Recip` kernel may lead to undefined behavior.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor::shape_type shape = {2, 3};
         * nz::data::MappedTensor tensor(shape, true);
         * tensor.recip();
         * ```
         * @endcode
         */
        void recip();

        /**
         * @brief Calculate the sum of all elements in the MappedTensor.
         *
         * @return The sum of all elements in the MappedTensor as a value of type `MappedTensor::value_type`.
         *
         * This function computes the sum of all elements within the MappedTensor. It utilizes CUDA parallel processing to perform the summation efficiently. First, it determines the block and grid dimensions for the CUDA kernel. Then, it allocates pinned host memory for storing the intermediate results using `cudaMallocHost`. The `krnl::Summation` CUDA kernel is launched to calculate partial sums on the device. After the kernel execution, the function synchronizes the device using `cudaDeviceSynchronize` to ensure that all operations are completed. Finally, it sums up the partial results on the host, frees the allocated pinned host memory, and returns the total sum.
         *
         * Memory management:
         * - Pinned host memory is allocated for `dData` using `cudaMallocHost` and freed using `cudaFreeHost`.
         *
         * Exception handling:
         * - The `CHECK` macro is used to handle CUDA API errors. If any CUDA API call fails, the `CHECK` macro will throw an exception, causing the function to terminate.
         *
         * Relationship with other components:
         * - This function relies on the `krnl::Summation` CUDA kernel to perform partial sums on the device.
         * - It also depends on the `CHECK` macro to handle CUDA API errors and `cudaDeviceSynchronize` for device synchronization.
         *
         * @throws [Exception type thrown by CHECK macro] If there are CUDA API errors during memory allocation, kernel execution, or memory synchronization.
         *
         * @note
         * - The time complexity of this function is approximately O(n), where n is the number of elements in the MappedTensor (`_size`). The CUDA kernel parallelizes the partial sum calculation, and the final sum on the host is a linear operation over the number of grid blocks.
         * - Ensure that the CUDA device is properly initialized before calling this function.
         * - Pinned host memory allocation may have limitations, so be aware of potential memory constraints.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor mapped_tensor({2, 3}, true);
         * // Assume mapped_tensor is filled with some values
         * nz::data::MappedTensor::value_type sum_result = mapped_tensor.sum();
         * ```
         * @endcode
         */
        [[nodiscard]] value_type sum() const;

        /**
         * @brief Calculate the sum of the exponential values of all elements in the MappedTensor.
         *
         * @return The sum of the exponential values of all elements in the MappedTensor as a value of type `MappedTensor::value_type`.
         *
         * This function computes the sum of the exponential values of all elements within the MappedTensor. It first determines the CUDA block and grid dimensions based on the size of the tensor. Then, it allocates pinned host memory using `cudaMallocHost` to store the intermediate results. The `krnl::SummationExp` CUDA kernel is launched to calculate the partial sums of the exponential values on the device. After the kernel execution, the function synchronizes the device using `cudaDeviceSynchronize` to ensure all operations are completed. Finally, it sums up the partial results on the host, frees the allocated pinned host memory, and returns the total sum.
         *
         * Memory management:
         * - Pinned host memory is allocated for `dData` using `cudaMallocHost` and freed using `cudaFreeHost`.
         *
         * Exception handling:
         * - The `CHECK` macro is used to handle CUDA API errors. If any CUDA API call fails, the `CHECK` macro will throw an exception, causing the function to terminate.
         *
         * Relationship with other components:
         * - This function relies on the `krnl::SummationExp` CUDA kernel to perform partial sums of exponential values on the device.
         * - It also depends on the `CHECK` macro to handle CUDA API errors and `cudaDeviceSynchronize` for device synchronization.
         *
         * @throws [Exception type thrown by CHECK macro] If there are CUDA API errors during memory allocation, kernel execution, or memory synchronization.
         *
         * @note
         * - The time complexity of this function is approximately O(n), where n is the number of elements in the MappedTensor (`_size`). The CUDA kernel parallelizes the partial sum calculation of exponential values, and the final sum on the host is a linear operation over the number of grid blocks.
         * - Ensure that the CUDA device is properly initialized before calling this function.
         * - Pinned host memory allocation may have limitations, so be aware of potential memory constraints.
         *
         * @code
         * ```cpp
         * nz::data::MappedTensor mapped_tensor({2, 3}, true);
         * // Assume mapped_tensor is filled with some values
         * nz::data::MappedTensor::value_type exp_sum_result = mapped_tensor.expSum();
         * ```
         * @endcode
         */
        [[nodiscard]] value_type expSum() const;

        /**
         * @brief Synchronizes the gradient data if gradient computation is required.
         *
         * This function checks the `_requires_grad` flag. If the flag is set to `true`, it calls the `syncData` method of the `cuStrm::streamManagerFP32` object, passing the `_grad` data. The `syncData` method blocks the host until all CUDA stream write operations on the input data are completed.
         *
         * @return None
         *
         * There is no explicit memory allocation or deallocation in this function. Memory management for the `_grad` data is assumed to be handled elsewhere.
         * The function does not have an explicit exception - handling mechanism. It relies on the `cuStrm::streamManagerFP32.syncData` method to manage any errors during the synchronization process.
         *
         * @note
         * - The time complexity of this function depends on the time it takes for the CUDA stream write operations on `_grad` to complete. In the worst - case scenario, if there are long - running write operations, it could take a significant amount of time.
         *
         * @code
         * ```cpp
         * // Assume MappedTensor is defined and an instance is created
         * MappedTensor mappedTensor;
         * mappedTensor.syncGrad();
         * ```
         * @endcode
         */
        void syncGrad() const;

        /**
         * @brief Synchronizes the tensor data by waiting for all CUDA stream write operations on it to finish.
         *
         * This function invokes the `syncData` method of the `cuStrm::streamManagerFP32` object, passing the `_data` member of the `MappedTensor` class. It blocks the host until all CUDA stream write operations on the `_data` are completed.
         *
         * @param None
         *
         * @return None
         *
         * Memory management for the `_data` is assumed to be handled elsewhere. There is no memory allocation or deallocation within this function.
         * This function does not have an explicit exception - handling mechanism. It depends on the `cuStrm::streamManagerFP32.syncData` method to handle any errors during the synchronization process.
         *
         * @note
         * - The time complexity of this function depends on the time taken for the CUDA stream write operations on `_data` to complete. In the worst - case scenario, it could take a long time if there are long - running write operations.
         *
         * @code
         * ```cpp
         * // Assume MappedTensor is defined and an instance is created
         * MappedTensor mappedTensor;
         * mappedTensor.syncData();
         * ```
         * @endcode
         */
        void syncData() const;

        /**
         * @brief Synchronizes the tensor data and its gradient.
         *
         * This function first calls the `syncData` method of the `cuStrm::streamManagerFP32` object, passing the `_data` member of the `MappedTensor` class. This is to ensure that all CUDA stream write operations on the tensor data are completed by blocking the host. Then it calls the `syncGrad` method to synchronize the gradient data if gradient computation is required.
         *
         * @param None
         *
         * @return None
         *
         * There is no explicit memory allocation or deallocation in this function. Memory management for the `_data` and `_grad` data is assumed to be handled elsewhere.
         * The function does not have an explicit exception - handling mechanism. It relies on the `cuStrm::streamManagerFP32.syncData` method and the `syncGrad` method to manage any errors during the synchronization process.
         *
         * @note
         * - The time complexity of this function depends on the time it takes for the CUDA stream write operations on `_data` and `_grad` (if applicable) to complete. In the worst - case scenario, if there are long - running write operations, it could take a significant amount of time.
         *
         * @code
         * ```cpp
         * // Assume MappedTensor is defined and an instance is created
         * MappedTensor mappedTensor;
         * mappedTensor.sync();
         * ```
         * @endcode
         */
        void sync() const;
        /// @}

    private:
        size_type _size;
        shape_type _shape;
        value_type* _data;
        value_type* _grad;
        bool _requires_grad;
    };
}
#endif //MAPPEDTENSOR_CUH

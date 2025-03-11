#ifndef TENSOROPERATIONS_CUH
#define TENSOROPERATIONS_CUH
#include "dl_export.cuh"
#include "Tensor.cuh"
#include "MappedTensor.cuh"
#define BLOCKSIZE 512

namespace nz::data {
    template <typename T>
    struct is_valid_tensor_type : std::disjunction<
            std::is_same<T, Tensor>,
            std::is_same<T, MappedTensor>
        > {
    };


    DL_API void iRELU(float* output, const float* input, unsigned long long size);

    /**
     * @brief Apply the Rectified Linear Unit (ReLU) activation function element-wise to an input tensor.
     *
     * @param input The input tensor (either `Tensor` or `MappedTensor`) to which the ReLU function will be applied (device to device).
     *
     * @return A new tensor (of the same type as the input: `Tensor` or `MappedTensor`) with the ReLU function applied element-wise.
     *
     * This function applies the ReLU activation function, defined as \( f(x) = \max(0, x) \), to each element of the input tensor. It first creates a new tensor `result` with the same shape and gradient requirement as the input tensor. Then, it calls the `iRELU` function to perform the actual ReLU operation on the data of the input tensor and store the results in the `result` tensor. Finally, the `result` tensor is returned.
     *
     * Memory management: A new tensor `result` is created, and its memory is managed by the tensor's own class (`Tensor` or `MappedTensor`). The memory of the input tensor remains unchanged.
     * Exception handling: There is no explicit exception handling in this function. However, if the `iRELU` function or the tensor constructors throw exceptions, they will propagate up.
     * Relationship with other components: This function depends on the `iRELU` function to perform the ReLU operation and the tensor's constructor to create a new tensor.
     *
     * @throws [Exception type thrown by iRELU or tensor constructors] If there are issues during the operation, such as memory allocation failures or incorrect input data.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the number of elements in the input tensor (`input.size()`), as it needs to apply the ReLU function to each element.
     *
     * @code
     * ```cpp
     * // Assume T is either Tensor or MappedTensor
     * nz::data::T::shape_type shape = {2, 3};
     * nz::data::T input(shape, true);
     * nz::data::T output = ReLU(input);
     * ```
     * @endcode
     */
    template <typename T>
    T ReLU(T input) {
        T result(input.shape(), input.requiresGrad());
        iRELU(result.data(), input.data(), input.size());
        return result;
    }

    DL_API void iSigmoid(float* output, const float* input, unsigned long long size);

    /**
     * @brief Apply the sigmoid activation function element-wise to an input tensor.
     *
     * @param input The input tensor (either `Tensor` or `MappedTensor`) to which the sigmoid function will be applied (device-to-device).
     *
     * @return A new tensor (of the same type as the input: `Tensor` or `MappedTensor`) with the sigmoid function applied element-wise.
     *
     * This function applies the sigmoid activation function, defined as \( f(x)=\frac{1}{1 + e^{-x}} \), to each element of the input tensor. It first creates a new tensor `result` with the same shape and gradient requirement as the input tensor. Then, it calls the `iSigmoid` function to perform the actual sigmoid operation on the data of the input tensor and store the results in the `result` tensor. Finally, the `result` tensor is returned.
     *
     * Memory management: A new tensor `result` is created, and its memory is managed by the tensor's own class (`Tensor` or `MappedTensor`). The memory of the input tensor remains unchanged.
     * Exception handling: There is no explicit exception handling in this function. However, if the `iSigmoid` function or the tensor constructors throw exceptions, they will propagate up.
     * Relationship with other components: This function depends on the `iSigmoid` function to perform the sigmoid operation and the tensor's constructor to create a new tensor.
     *
     * @throws [Exception type thrown by iSigmoid or tensor constructors] If there are issues during the operation, such as memory allocation failures or incorrect input data.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the number of elements in the input tensor (`input.size()`), as it needs to apply the sigmoid function to each element.
     *
     *
     * @code
     * ```cpp
     * // Assume T is either Tensor or MappedTensor
     * nz::data::T::shape_type shape = {2, 3};
     * nz::data::T input(shape, true);
     * nz::data::T output = Sigmoid(input);
     * ```
     * @endcode
     */
    template <typename T>
    T Sigmoid(T input) {
        T result(input.shape(), input.requiresGrad());
        iSigmoid(result.data(), input.data(), input.size());
        return result;
    }

    DL_API void iTanh(float* output, const float* input, unsigned long long size);

    /**
     * @brief Apply the hyperbolic tangent (tanh) activation function element-wise to an input tensor.
     *
     * @param input The input tensor (either `Tensor` or `MappedTensor`) to which the tanh function will be applied (device-to-device).
     *
     * @return A new tensor (of the same type as the input: `Tensor` or `MappedTensor`) with the tanh function applied element-wise.
     *
     * This function applies the hyperbolic tangent activation function, defined as \( f(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} \), to each element of the input tensor. It first creates a new tensor `result` with the same shape and gradient requirement as the input tensor. Then, it calls the `iTanh` function to perform the actual tanh operation on the data of the input tensor and store the results in the `result` tensor. Finally, the `result` tensor is returned.
     *
     * Memory management: A new tensor `result` is created, and its memory is managed by the tensor's own class (`Tensor` or `MappedTensor`). The memory of the input tensor remains unchanged.
     * Exception handling: There is no explicit exception handling in this function. However, if the `iTanh` function or the tensor constructors throw exceptions, they will propagate up.
     * Relationship with other components: This function depends on the `iTanh` function to perform the tanh operation and the tensor's constructor to create a new tensor.
     *
     * @throws [Exception type thrown by iTanh or tensor constructors] If there are issues during the operation, such as memory allocation failures or incorrect input data.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the number of elements in the input tensor (`input.size()`), as it needs to apply the tanh function to each element.
     *
     *
     * @code
     * ```cpp
     * // Assume T is either Tensor or MappedTensor
     * nz::data::T::shape_type shape = {2, 3};
     * nz::data::T input(shape, true);
     * nz::data::T output = Tanh(input);
     * ```
     * @endcode
     */
    template <typename T>
    T Tanh(T input) {
        T result(input.shape(), input.requiresGrad());
        iTanh(result.data(), input.data(), input.size());
        return result;
    }

    DL_API void iLeakyReLU(float* output, const float* input, unsigned long long size, float alpha);

    /**
     * @brief Apply the Leaky Rectified Linear Unit (Leaky ReLU) activation function element-wise to an input tensor.
     *
     * @param input The input tensor (either `Tensor` or `MappedTensor`) to which the Leaky ReLU function will be applied (device-to-device).
     * @param alpha The slope coefficient for negative values. It has a default value of 0.01f.
     *
     * @return A new tensor (of the same type as the input: `Tensor` or `MappedTensor`) with the Leaky ReLU function applied element-wise.
     *
     * This function applies the Leaky ReLU activation function, defined as \( f(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases} \), to each element of the input tensor. It first creates a new tensor `result` with the same shape and gradient requirement as the input tensor. Then, it calls the `iLeakyReLU` function to perform the actual Leaky ReLU operation on the data of the input tensor and store the results in the `result` tensor. Finally, the `result` tensor is returned.
     *
     * Memory management: A new tensor `result` is created, and its memory is managed by the tensor's own class (`Tensor` or `MappedTensor`). The memory of the input tensor remains unchanged.
     * Exception handling: There is no explicit exception handling in this function. However, if the `iLeakyReLU` function or the tensor constructors throw exceptions, they will propagate up.
     * Relationship with other components: This function depends on the `iLeakyReLU` function to perform the Leaky ReLU operation and the tensor's constructor to create a new tensor.
     *
     * @throws [Exception type thrown by iLeakyReLU or tensor constructors] If there are issues during the operation, such as memory allocation failures or incorrect input data.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the number of elements in the input tensor (`input.size()`), as it needs to apply the Leaky ReLU function to each element.
     * - The value of `alpha` should be a small positive number to avoid vanishing gradient problem for negative inputs.
     *
     * @code
     * ```cpp
     * // Assume T is either Tensor or MappedTensor
     * nz::data::T::shape_type shape = {2, 3};
     * nz::data::T input(shape, true);
     * nz::data::T output = LeakyReLU(input, 0.02f);
     * ```
     * @endcode
     */
    template <typename T>
    T LeakyReLU(T input, const float alpha = 0.01f) {
        T result(input.shape(), input.requiresGrad());
        iLeakyReLU(result.data(), input.data(), input.size(), alpha);
        return result;
    }

    DL_API void iSwish(float* output, const float* input, unsigned long long size);

    /**
     * @brief Apply the Swish activation function element-wise to an input tensor.
     *
     * @param input The input tensor (either `Tensor` or `MappedTensor`) to which the Swish function will be applied (device-to-device).
     *
     * @return A new tensor (of the same type as the input: `Tensor` or `MappedTensor`) with the Swish function applied element-wise.
     *
     * This function applies the Swish activation function, defined as \( f(x)=x\cdot\sigma(x) \), where \(\sigma(x)=\frac{1}{1 + e^{-x}}\) is the sigmoid function, to each element of the input tensor. It first creates a new tensor `result` with the same shape and gradient requirement as the input tensor. Then, it calls the `iSwish` function to perform the actual Swish operation on the data of the input tensor and store the results in the `result` tensor. Finally, the `result` tensor is returned.
     *
     * Memory management: A new tensor `result` is created, and its memory is managed by the tensor's own class (`Tensor` or `MappedTensor`). The memory of the input tensor remains unchanged.
     * Exception handling: There is no explicit exception handling in this function. However, if the `iSwish` function or the tensor constructors throw exceptions, they will propagate up.
     * Relationship with other components: This function depends on the `iSwish` function to perform the Swish operation and the tensor's constructor to create a new tensor.
     *
     * @throws [Exception type thrown by iSwish or tensor constructors] If there are issues during the operation, such as memory allocation failures or incorrect input data.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the number of elements in the input tensor (`input.size()`), as it needs to apply the Swish function to each element.
     *
     * @code
     * ```cpp
     * // Assume T is either Tensor or MappedTensor
     * nz::data::T::shape_type shape = {2, 3};
     * nz::data::T input(shape, true);
     * nz::data::T output = Swish(input);
     * ```
     * @endcode
     */
    template <typename T>
    T Swish(T input) {
        T result(input.shape(), input.requiresGrad());
        iSwish(result.data(), input.data(), input.size());
        return result;
    }

    DL_API void iELU(float* output, const float* input, unsigned long long size, float alpha);

    /**
     * @brief Apply the Exponential Linear Unit (ELU) activation function element-wise to an input tensor.
     *
     * @param input The input tensor (either `Tensor` or `MappedTensor`) to which the ELU function will be applied (device-to-device).
     * @param alpha The alpha value for the ELU function. It controls the value to which the function saturates for negative inputs. The default value is 1.0f.
     *
     * @return A new tensor (of the same type as the input: `Tensor` or `MappedTensor`) with the ELU function applied element-wise.
     *
     * This function applies the ELU activation function, defined as \( f(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha (e^{x}- 1) & \text{if } x < 0 \end{cases} \), to each element of the input tensor. It first creates a new tensor `result` with the same shape and gradient requirement as the input tensor. Then, it calls the `iELU` function to perform the actual ELU operation on the data of the input tensor and store the results in the `result` tensor. Finally, the `result` tensor is returned.
     *
     * Memory management: A new tensor `result` is created, and its memory is managed by the tensor's own class (`Tensor` or `MappedTensor`). The memory of the input tensor remains unchanged.
     * Exception handling: There is no explicit exception handling in this function. However, if the `iELU` function or the tensor constructors throw exceptions, they will propagate up.
     * Relationship with other components: This function depends on the `iELU` function to perform the ELU operation and the tensor's constructor to create a new tensor.
     *
     * @throws [Exception type thrown by iELU or tensor constructors] If there are issues during the operation, such as memory allocation failures or incorrect input data.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the number of elements in the input tensor (`input.size()`), as it needs to apply the ELU function to each element.
     * - A positive `alpha` value is recommended for better performance and to avoid the vanishing gradient problem.
     *
     * @code
     * ```cpp
     * // Assume T is either Tensor or MappedTensor
     * nz::data::T::shape_type shape = {2, 3};
     * nz::data::T input(shape, true);
     * nz::data::T output = ELU(input, 0.5f);
     * ```
     * @endcode
     */
    template <typename T>
    T ELU(T input, const float alpha = 1.0f) {
        T result(input.shape(), input.requiresGrad());
        iELU(result.data(), input.data(), input.size(), alpha);
        return result;
    }

    DL_API void iHardSigmoid(float* output, const float* input, unsigned long long size, float alpha, float beta);

    /**
     * @brief Apply the Hard Sigmoid activation function element-wise to an input tensor.
     *
     * @param input The input tensor (either `Tensor` or `MappedTensor`) to which the Hard Sigmoid function will be applied (device-to-device).
     * @param alpha The alpha value for the Hard Sigmoid function, controlling the slope of the linear part. The default value is 0.2f.
     * @param beta The beta value for the Hard Sigmoid function, controlling the bias of the linear part. The default value is 0.5f.
     *
     * @return A new tensor (of the same type as the input: `Tensor` or `MappedTensor`) with the Hard Sigmoid function applied element-wise.
     *
     * This function applies the Hard Sigmoid activation function, typically defined as \( f(x) = \max(0, \min(1, \alpha x + \beta)) \), to each element of the input tensor. It first creates a new tensor `result` with the same shape and gradient requirement as the input tensor. Then, it calls the `iHardSigmoid` function to perform the actual Hard Sigmoid operation on the data of the input tensor and store the results in the `result` tensor. Finally, the `result` tensor is returned.
     *
     * Memory management: A new tensor `result` is created, and its memory is managed by the tensor's own class (`Tensor` or `MappedTensor`). The memory of the input tensor remains unchanged.
     * Exception handling: There is no explicit exception handling in this function. However, if the `iHardSigmoid` function or the tensor constructors throw exceptions, they will propagate up.
     * Relationship with other components: This function depends on the `iHardSigmoid` function to perform the Hard Sigmoid operation and the tensor's constructor to create a new tensor.
     *
     * @throws [Exception type thrown by iHardSigmoid or tensor constructors] If there are issues during the operation, such as memory allocation failures or incorrect input data.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the number of elements in the input tensor (`input.size()`), as it needs to apply the Hard Sigmoid function to each element.
     * - The choice of `alpha` and `beta` values can significantly affect the behavior of the Hard Sigmoid function.
     *
     * @code
     * ```cpp
     * // Assume T is either Tensor or MappedTensor
     * nz::data::T::shape_type shape = {2, 3};
     * nz::data::T input(shape, true);
     * nz::data::T output = HardSigmoid(input, 0.3f, 0.6f);
     * ```
     * @endcode
     */
    template <typename T>
    T HardSigmoid(T input, const float alpha = 0.2f, const float beta = 0.5f) {
        T result(input.shape(), input.requiresGrad());
        iHardSigmoid(result.data(), input.data(), input.size(), alpha, beta);
        return result;
    }

    DL_API void iHardSwish(float* output, const float* input, unsigned long long size, float alpha, float beta);

    /**
     * @brief Apply the Hard Swish activation function element-wise to an input tensor.
     *
     * @param input The input tensor (either `Tensor` or `MappedTensor`) to which the Hard Swish function will be applied (device-to-device).
     * @param alpha The alpha value for the Hard Swish function, used to scale the input. The default value is 0.5f.
     * @param beta The beta value for the Hard Swish function, used as an offset. The default value is 0.5f.
     *
     * @return A new tensor (of the same type as the input: `Tensor` or `MappedTensor`) with the Hard Swish function applied element-wise.
     *
     * This function applies the Hard Swish activation function to each element of the input tensor. The Hard Swish function is often defined as \( f(x)=x \cdot \max(0, \min(1, \alpha x+\beta)) \). It first creates a new tensor `result` with the same shape and gradient requirement as the input tensor. Then, it calls the `iHardSwish` function to perform the actual Hard Swish operation on the data of the input tensor and store the results in the `result` tensor. Finally, the `result` tensor is returned.
     *
     * Memory management: A new tensor `result` is created, and its memory is managed by the tensor's own class (`Tensor` or `MappedTensor`). The memory of the input tensor remains unchanged.
     * Exception handling: There is no explicit exception handling in this function. However, if the `iHardSwish` function or the tensor constructors throw exceptions, they will propagate up.
     * Relationship with other components: This function depends on the `iHardSwish` function to perform the Hard Swish operation and the tensor's constructor to create a new tensor.
     *
     * @throws [Exception type thrown by iHardSwish or tensor constructors] If there are issues during the operation, such as memory allocation failures or incorrect input data.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the number of elements in the input tensor (`input.size()`), as it needs to apply the Hard Swish function to each element.
     * - The values of `alpha` and `beta` can be adjusted to fine - tune the behavior of the Hard Swish function.
     *
     * @code
     * ```cpp
     * // Assume T is either Tensor or MappedTensor
     * nz::data::T::shape_type shape = {2, 3};
     * nz::data::T input(shape, true);
     * nz::data::T output = HardSwish(input, 0.4f, 0.7f);
     * ```
     * @endcode
     */
    template <typename T>
    T HardSwish(T input, const float alpha = 0.5f, const float beta = 0.5f) {
        T result(input.shape(), input.requiresGrad());
        iHardSwish(result.data(), input.data(), input.size(), alpha, beta);
        return result;
    }

    DL_API void iSoftmax(float* output, const float* input, float sum, unsigned long long size);

    /**
     * @brief Compute the softmax function for a given input of type T.
     *
     * @param input The input object of type T for which the softmax function will be computed. The input is passed by value, so a copy of the input is made inside the function.
     *
     * @return An object of type T representing the result of the softmax function applied to the input.
     *
     * This function computes the softmax function for the given input. It first creates a new object `result` with the same shape and gradient requirement as the input. Then, it calls the `iSoftmax` function to perform the actual softmax computation. The `iSoftmax` function takes the data pointers of the result and input, the exponential sum of the input, and the size of the input as parameters. Finally, the computed result is returned.
     *
     * Memory management:
     * - A new object `result` is created inside the function, which may allocate memory depending on the implementation of the constructor of type T. The memory for the result will be managed by the destructor of the object when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. However, if the `iSoftmax` function or the constructor of type T throws an exception, it will propagate up to the caller.
     *
     * Relationship with other components:
     * - This function depends on the `iSoftmax` function to perform the actual softmax computation.
     * - It also depends on the `shape()`, `requiresGrad()`, `expSum()`, and `size()` member functions of type T.
     *
     * @note
     * - The time complexity of this function depends on the implementation of the `iSoftmax` function. If the `iSoftmax` function has a time complexity of O(n), where n is the size of the input, then the overall time complexity of this function is also O(n).
     * - Ensure that the input object `input` has valid shape, gradient requirement, exponential sum, and size information.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid type with shape(), requiresGrad(), expSum(), and size() member functions
     * nz::data::Tensor input({2, 3}, true);
     * // Assume input is filled with some values
     * nz::data::Tensor result = Softmax(input);
     * ```
     * @endcode
     */
    template <typename T>
    T Softmax(T input) {
        T result(input.shape(), input.requiresGrad());
        iSoftmax(result.data(), input.data(), input.expSum(), input.size());
        return result;
    }

    DL_API void iScalarAdd(float* output, const float* input, float scalar, unsigned long long size);

    DL_API void iScalarDiv(float* output, const float* input, float scalar, unsigned long long size);

    DL_API void iScalarMul(float* output, const float* input, float scalar, unsigned long long size);

    /**
     * @brief Overload the addition operator to add a scalar float to a tensor of type T.
     *
     * @param lhs A reference to the left - hand side tensor of type T. The tensor data is modified in - place during the addition operation.
     * @param rhs A constant float value representing the right - hand side scalar to be added to the tensor.
     *
     * @return A new tensor of type T that is the result of adding the scalar rhs to each element of the tensor lhs.
     *
     * This function is a template operator overload that adds a scalar float value to a tensor. It first checks if the type T meets the requirements using `is_valid_tensor_type<T>::value`. If the type is valid, it creates a new tensor `result` with the same shape and gradient requirement as `lhs`. Then, it calls the `iScalarAdd` function to perform the actual addition operation, which adds the scalar `rhs` to each element of the data in `lhs` and stores the result in `result`. Finally, the newly created tensor `result` is returned.
     *
     * Memory management:
     * - A new tensor `result` is created inside the function, which may allocate memory depending on the implementation of the constructor of type T. The memory for the result will be managed by the destructor of the object when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. However, if the `iScalarAdd` function or the constructor of type T throws an exception, it will propagate up to the caller.
     *
     * Relationship with other components:
     * - This function depends on the `iScalarAdd` function to perform the actual scalar - tensor addition.
     * - It also depends on the `shape()` and `requiresGrad()` member functions of type T.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor `lhs`. This is because the `iScalarAdd` function needs to iterate over each element of the tensor.
     * - Ensure that the type T is a valid tensor type as determined by `is_valid_tensor_type<T>::value`.
     * - Ensure that the tensor `lhs` has valid shape, gradient requirement, and size information.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid tensor type with shape(), requiresGrad() member functions
     * nz::data::Tensor tensor({2, 3}, true);
     * // Assume tensor is filled with some values
     * nz::data::Tensor result = tensor + 2.0f;
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    operator+(T& lhs, const float rhs) {
        T result(lhs.shape(), lhs.requiresGrad());
        iScalarAdd(result.data(), lhs.data(), rhs, lhs.size());
        return result;
    }

    /**
     * @brief Overload the addition operator to add a tensor of type T to a scalar float.
     *
     * @param lhs A constant float value representing the left - hand side scalar to be added to the tensor.
     * @param rhs A reference to the right - hand side tensor of type T. The tensor data is used to perform the addition operation.
     *
     * @return A new tensor of type T that is the result of adding the scalar lhs to each element of the tensor rhs.
     *
     * This function is a template operator overload. It first checks if the type T is a valid tensor type using `is_valid_tensor_type<T>::value`. If the type is valid, it creates a new tensor `result` with the same shape and gradient requirement as `rhs`. Then, it calls the `iScalarAdd` function to add the scalar `lhs` to each element of the data in `rhs` and stores the result in `result`. Finally, the newly created tensor `result` is returned.
     *
     * Memory management:
     * - A new tensor `result` is created inside the function, which may allocate memory according to the constructor of type T. The memory of the result will be managed by its destructor when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. If the `iScalarAdd` function or the constructor of type T throws an exception, it will be propagated to the caller.
     *
     * Relationship with other components:
     * - This function depends on the `iScalarAdd` function to perform the actual scalar - tensor addition.
     * - It also depends on the `shape()` and `requiresGrad()` member functions of type T.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor `rhs`. This is because the `iScalarAdd` function needs to iterate over each element of the tensor.
     * - Ensure that the type T is a valid tensor type as determined by `is_valid_tensor_type<T>::value`.
     * - Ensure that the tensor `rhs` has valid shape, gradient requirement, and size information.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid tensor type with shape(), requiresGrad() member functions
     * nz::data::Tensor tensor({2, 3}, true);
     * // Assume tensor is filled with some values
     * nz::data::Tensor result = 2.0f + tensor;
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    operator+(const float lhs, T& rhs) {
        T result(rhs.shape(), rhs.requiresGrad());
        iScalarAdd(result.data(), rhs.data(), lhs, rhs.size());
        return result;
    }

    /**
     * @brief Overload the subtraction operator to subtract a scalar float from a tensor of type T.
     *
     * @param lhs A reference to the left - hand side tensor of type T. The tensor data is used as the base for the subtraction operation.
     * @param rhs A constant float value representing the right - hand side scalar to be subtracted from the tensor.
     *
     * @return A new tensor of type T that is the result of subtracting the scalar rhs from each element of the tensor lhs.
     *
     * This template operator overload first checks if the type T is a valid tensor type using `is_valid_tensor_type<T>::value`. If valid, it creates a new tensor `result` with the same shape and gradient requirement as `lhs`. To perform the subtraction, it calls the `iScalarAdd` function with `-rhs` as the scalar to be added to each element of `lhs` data. Finally, the newly created tensor `result` is returned.
     *
     * Memory management:
     * - A new tensor `result` is created inside the function, which may allocate memory based on the constructor of type T. The memory of the result will be managed by its destructor when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. If the `iScalarAdd` function or the constructor of type T throws an exception, it will propagate to the caller.
     *
     * Relationship with other components:
     * - This function depends on the `iScalarAdd` function to perform the actual subtraction operation (by adding the negative of the scalar).
     * - It also depends on the `shape()` and `requiresGrad()` member functions of type T.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor `lhs`. This is because the `iScalarAdd` function needs to iterate over each element of the tensor.
     * - Ensure that the type T is a valid tensor type as determined by `is_valid_tensor_type<T>::value`.
     * - Ensure that the tensor `lhs` has valid shape, gradient requirement, and size information.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid tensor type with shape(), requiresGrad() member functions
     * nz::data::Tensor tensor({2, 3}, true);
     * // Assume tensor is filled with some values
     * nz::data::Tensor result = tensor - 2.0f;
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    operator-(T& lhs, const float rhs) {
        T result(lhs.shape(), lhs.requiresGrad());
        iScalarAdd(result.data(), lhs.data(), -rhs, lhs.size());
        return result;
    }

    /**
     * @brief Overload the subtraction operator to subtract a tensor of type T from a scalar float.
     *
     * @param lhs A constant float value representing the left - hand side scalar from which the tensor will be subtracted.
     * @param rhs A reference to the right - hand side tensor of type T. The tensor data is used in the subtraction operation.
     *
     * @return A new tensor of type T that is the result of subtracting each element of the tensor rhs from the scalar lhs.
     *
     * This template operator overload first checks if the type T is a valid tensor type using `is_valid_tensor_type<T>::value`. If the type is valid, it creates a new tensor `result` by negating the tensor `rhs`. Then, it calls the `iScalarAdd` function to add the scalar `lhs` to each element of the negated tensor `result`. Finally, the resulting tensor `result` is returned.
     *
     * Memory management:
     * - A new tensor `result` is created inside the function, which may allocate memory according to the constructor of type T. The memory of the result will be managed by its destructor when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. If the negation operation of `rhs`, the `iScalarAdd` function, or the constructor of type T throws an exception, it will be propagated to the caller.
     *
     * Relationship with other components:
     * - This function depends on the negation operator of type T to obtain the negated tensor.
     * - It also depends on the `iScalarAdd` function to perform the addition of the scalar to the negated tensor.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor `rhs`. This is because both the negation operation and the `iScalarAdd` function need to iterate over each element of the tensor.
     * - Ensure that the type T is a valid tensor type as determined by `is_valid_tensor_type<T>::value`.
     * - Ensure that the tensor `rhs` has valid shape, gradient requirement, and size information.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid tensor type
     * nz::data::Tensor tensor({2, 3}, true);
     * // Assume tensor is filled with some values
     * nz::data::Tensor result = 2.0f - tensor;
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    operator-(const float lhs, T& rhs) {
        T result = -rhs;
        iScalarAdd(result.data(), result.data(), lhs, result.size());
        return result;
    }

    /**
     * @brief Overload the multiplication operator to multiply a tensor of type T by a scalar float.
     *
     * @param lhs A reference to the left - hand side tensor of type T. The tensor data is used as the base for the multiplication operation.
     * @param rhs A constant float value representing the right - hand side scalar to multiply the tensor by.
     *
     * @return A new tensor of type T that is the result of multiplying each element of the tensor lhs by the scalar rhs.
     *
     * This template operator overload first checks if the type T is a valid tensor type using `is_valid_tensor_type<T>::value`. If valid, it creates a new tensor `result` with the same shape and gradient requirement as `lhs`. To perform the multiplication, it calls the `iScalarMul` function to multiply each element of `lhs` data by the scalar `rhs`. Finally, the newly created tensor `result` is returned.
     *
     * Memory management:
     * - A new tensor `result` is created inside the function, which may allocate memory based on the constructor of type T. The memory of the result will be managed by its destructor when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. If the `iScalarMul` function or the constructor of type T throws an exception, it will propagate to the caller.
     *
     * Relationship with other components:
     * - This function depends on the `iScalarMul` function to perform the actual multiplication operation.
     * - It also depends on the `shape()` and `requiresGrad()` member functions of type T.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor `lhs`. This is because the `iScalarMul` function needs to iterate over each element of the tensor.
     * - Ensure that the type T is a valid tensor type as determined by `is_valid_tensor_type<T>::value`.
     * - Ensure that the tensor `lhs` has valid shape, gradient requirement, and size information.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid tensor type with shape(), requiresGrad() member functions
     * nz::data::Tensor tensor({2, 3}, true);
     * // Assume tensor is filled with some values
     * nz::data::Tensor result = tensor * 2.0f;
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    operator*(T& lhs, const float rhs) {
        T result(lhs.shape(), lhs.requiresGrad());
        iScalarMul(result.data(), lhs.data(), rhs, lhs.size());
        return result;
    }

    /**
     * @brief Overload the multiplication operator to multiply a scalar float by a tensor of type T.
     *
     * @param lhs A constant float value representing the left - hand side scalar to multiply the tensor by.
     * @param rhs A reference to the right - hand side tensor of type T. The tensor data is used in the multiplication operation.
     *
     * @return A new tensor of type T that is the result of multiplying each element of the tensor rhs by the scalar lhs.
     *
     * This template operator overload first verifies if the type T is a valid tensor type using `is_valid_tensor_type<T>::value`. If the type is valid, it constructs a new tensor `result` with the same shape and gradient requirement as `rhs`. Subsequently, it invokes the `iScalarMul` function to multiply each element of `rhs` data by the scalar `lhs`. Finally, the newly created tensor `result` is returned.
     *
     * Memory management:
     * - A new tensor `result` is created within the function, and its memory allocation depends on the constructor of type T. The memory of `result` will be managed by its destructor when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. If the `iScalarMul` function or the constructor of type T throws an exception, it will be propagated to the caller.
     *
     * Relationship with other components:
     * - This function relies on the `iScalarMul` function to perform the actual multiplication operation.
     * - It also depends on the `shape()` and `requiresGrad()` member functions of type T.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor `rhs`. This is because the `iScalarMul` function needs to iterate over each element of the tensor.
     * - Ensure that the type T is a valid tensor type as determined by `is_valid_tensor_type<T>::value`.
     * - Ensure that the tensor `rhs` has valid shape, gradient requirement, and size information.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid tensor type with shape(), requiresGrad() member functions
     * nz::data::Tensor tensor({2, 3}, true);
     * // Assume tensor is filled with some values
     * nz::data::Tensor result = 2.0f * tensor;
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    operator*(const float lhs, T& rhs) {
        T result(rhs.shape(), rhs.requiresGrad());
        iScalarMul(result.data(), rhs.data(), lhs, rhs.size());
        return result;
    }

    /**
     * @brief Overload the division operator to divide a tensor of type T by a scalar float.
     *
     * @param lhs A reference to the left - hand side tensor of type T. The tensor data is used as the dividend for the division operation.
     * @param rhs A constant float value representing the right - hand side scalar divisor.
     *
     * @return A new tensor of type T that is the result of dividing each element of the tensor lhs by the scalar rhs.
     *
     * This template operator overload first checks if the type T is a valid tensor type using `is_valid_tensor_type<T>::value`. If valid, it creates a new tensor `result` with the same shape and gradient requirement as `lhs`. Then it calls the `iScalarDiv` function to divide each element of `lhs` data by the scalar `rhs`. Finally, the newly created tensor `result` is returned.
     *
     * Memory management:
     * - A new tensor `result` is created inside the function, and its memory allocation depends on the constructor of type T. The memory of `result` will be managed by its destructor when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. If the `iScalarDiv` function or the constructor of type T throws an exception, it will propagate to the caller.
     *
     * Relationship with other components:
     * - This function depends on the `iScalarDiv` function to perform the actual division operation.
     * - It also depends on the `shape()` and `requiresGrad()` member functions of type T.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor `lhs`. This is because the `iScalarDiv` function needs to iterate over each element of the tensor.
     * - Ensure that the type T is a valid tensor type as determined by `is_valid_tensor_type<T>::value`.
     * - Ensure that the tensor `lhs` has valid shape, gradient requirement, and size information.
     * - Ensure that the scalar `rhs` is not zero to avoid division by zero errors.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid tensor type with shape(), requiresGrad() member functions
     * nz::data::Tensor tensor({2, 3}, true);
     * // Assume tensor is filled with some values
     * nz::data::Tensor result = tensor / 2.0f;
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    operator/(T& lhs, const float rhs) {
        T result(lhs.shape(), lhs.requiresGrad());
        iScalarDiv(result.data(), lhs.data(), rhs, lhs.size());
        return result;
    }

    /**
     * @brief Overload the division operator to divide a scalar float by a tensor of type T.
     *
     * @param lhs A constant float value representing the left - hand side scalar dividend.
     * @param rhs A reference to the right - hand side tensor of type T. The tensor data is used as the divisor for the division operation.
     *
     * @return A new tensor of type T that is the result of dividing the scalar lhs by each element of the tensor rhs.
     *
     * This template operator overload first verifies if the type T is a valid tensor type using `is_valid_tensor_type<T>::value`. If valid, it creates a copy of the tensor `rhs` named `result`. Then it calls the `recip` method of `result` to compute the reciprocal of each element in the tensor. Finally, it uses the `iScalarMul` function to multiply each element of the `result` tensor by the scalar `lhs`.
     *
     * Memory management:
     * - A copy of the tensor `rhs` is created as `result`, and its memory allocation depends on the copy - constructor of type T. The memory of `result` will be managed by its destructor when it goes out of scope.
     *
     * Exception handling:
     * - There is no explicit exception handling in this function. If the `recip` method, `iScalarMul` function, or the copy - constructor of type T throws an exception, it will be propagated to the caller.
     *
     * Relationship with other components:
     * - This function depends on the `recip` method of type T to compute the reciprocal of each element in the tensor.
     * - It also depends on the `iScalarMul` function to perform the multiplication operation.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor `rhs`. This is because both the `recip` method and the `iScalarMul` function need to iterate over each element of the tensor.
     * - Ensure that the type T is a valid tensor type as determined by `is_valid_tensor_type<T>::value`.
     * - Ensure that the tensor `rhs` has valid shape, gradient requirement, and size information.
     * - Ensure that no element in the tensor `rhs` is zero to avoid division by zero errors during the `recip` operation.
     *
     * @code
     * ```cpp
     * // Assume Tensor is a valid tensor type with shape(), requiresGrad() and recip() member functions
     * nz::data::Tensor tensor({2, 3}, true);
     * // Assume tensor is filled with some non - zero values
     * nz::data::Tensor result = 2.0f / tensor;
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    operator/(const float lhs, T& rhs) {
        T result = rhs;
        result.recip();
        iScalarMul(result.data(), result.data(), lhs, result.size());
        return result;
    }
}
#endif //TENSOROPERATIONS_CUH

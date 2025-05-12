#ifndef TENSOROPERATIONS_CUH
#define TENSOROPERATIONS_CUH
#include "dl_export.cuh"
#include "Tensor.cuh"
#include "MappedTensor.cuh"
#include "OperationKernels.cuh"
#include "utils.cuh"
#define BLOCKSIZE 512

namespace nz::data {
    template <typename T>
    struct is_valid_tensor_type : std::disjunction<
            std::is_same<T, Tensor>,
            std::is_same<T, MappedTensor>
        > {
    };


    DL_API void iRELU(float* output, float* input, unsigned long long size);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    ReLU(T& input) {
        T result(input.shape(), input.requiresGrad());
        iRELU(result.data(), input.data(), input.size());
        return result;
    }

    DL_API void iSigmoid(float* output, float* input, unsigned long long size);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    Sigmoid(T& input) {
        T result(input.shape(), input.requiresGrad());
        iSigmoid(result.data(), input.data(), input.size());
        return result;
    }

    DL_API void iTanh(float* output, float* input, unsigned long long size);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    Tanh(T& input) {
        T result(input.shape(), input.requiresGrad());
        iTanh(result.data(), input.data(), input.size());
        return result;
    }

    DL_API void iLeakyReLU(float* output, float* input, unsigned long long size, float alpha);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    LeakyReLU(T& input, const float alpha = 0.01f) {
        T result(input.shape(), input.requiresGrad());
        iLeakyReLU(result.data(), input.data(), input.size(), alpha);
        return result;
    }

    DL_API void iSwish(float* output, float* input, unsigned long long size);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    Swish(T& input) {
        T result(input.shape(), input.requiresGrad());
        iSwish(result.data(), input.data(), input.size());
        return result;
    }

    DL_API void iELU(float* output, float* input, unsigned long long size, float alpha);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    ELU(T& input, const float alpha = 1.0f) {
        T result(input.shape(), input.requiresGrad());
        iELU(result.data(), input.data(), input.size(), alpha);
        return result;
    }

    DL_API void iHardSigmoid(float* output, float* input, unsigned long long size, float alpha, float beta);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    HardSigmoid(T& input, const float alpha = 0.2f, const float beta = 0.5f) {
        T result(input.shape(), input.requiresGrad());
        iHardSigmoid(result.data(), input.data(), input.size(), alpha, beta);
        return result;
    }

    DL_API void iHardSwish(float* output, float* input, unsigned long long size, float alpha, float beta);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    HardSwish(T& input, const float alpha = 0.5f, const float beta = 0.5f) {
        T result(input.shape(), input.requiresGrad());
        iHardSwish(result.data(), input.data(), input.size(), alpha, beta);
        return result;
    }

    DL_API void iSoftmax(float* output, float* input, const std::vector<float>& sum, unsigned long long size,
                         const std::vector<size_t>& offset);

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
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    Softmax(T& input) {
        T result(input.shape(), input.requiresGrad());
        auto size = input.shape()[2] * input.shape()[3];
        std::vector<size_t> offset;
        std::vector<float> sum;
        for (auto i = 0; i < input.shape()[0]; i++) {
            for (auto j = 0; j < input.shape()[1]; j++) {
                offset.push_back(i * input.shape().getStride(0) + j * input.shape().getStride(1));
                sum.push_back(input.expSum(i, j));
            }
        }
        iSoftmax(result.data(), input.data(), sum, size, offset);
        return result;
    }

    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, void>
    Softmax(T& output, T& input) {
        auto size = input.shape()[2] * input.shape()[3];
        std::vector<size_t> offset;
        std::vector<float> sum;
        for (auto i = 0; i < input.shape()[0]; i++) {
            for (auto j = 0; j < input.shape()[1]; j++) {
                offset.push_back(i * input.shape().getStride(0) + j * input.shape().getStride(1));
                sum.push_back(input.expSum(i, j));
            }
        }
        iSoftmax(output.data(), input.data(), sum, size, offset);
    }

    DL_API void iScalarAdd(float* output, float* input, float scalar, unsigned long long size);

    DL_API void iScalarDiv(float* output, float* input, float scalar, unsigned long long size);

    DL_API void iScalarMul(float* output, float* input, float scalar, unsigned long long size);

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

    DL_API void iMatrixAdd(float* out, float* in1, float* in2, size_t n, const std::vector<size_t>& offset_o,
                           const std::vector<size_t>& offset_i1, const std::vector<size_t>& offset_i2);

    /**
     * @brief Performs matrix addition operation on tensors with broadcast compatibility.
     *
     * This function is a template function that adds two tensors `lhs` and `rhs` and stores the result in `out`.
     * It only accepts tensor types for which `is_valid_tensor_type<T>::value` is `true`. The shapes of the input tensors
     * must be broadcast compatible, and the height and width dimensions must match.
     *
     * @tparam T The tensor type. This type must satisfy `is_valid_tensor_type<T>::value`.
     * @param out The output tensor where the result of the addition will be stored. Memory flow: host-to-function (for reference), function-to-host (modifies the object).
     * @param lhs The left-hand side tensor of the addition. Memory flow: host-to-function.
     * @param rhs The right-hand side tensor of the addition. Memory flow: host-to-function.
     *
     * @return None
     *
     * **Memory Management Strategy**:
     * - This function does not allocate or free any additional memory for the tensors. It only uses local `std::vector` objects (`offsetC`, `offsetA`, `offsetB`) to store offset values, and these vectors are automatically managed by their destructors.
     *
     * **Exception Handling Mechanism**:
     * - Throws `std::invalid_argument` if the shapes of `lhs` and `rhs` are not broadcast compatible or if their height and width dimensions do not match.
     *
     * **Relationship with Other Components**:
     * - Depends on the `shape()` method of the tensor type `T` to access shape information, including broadcast compatibility, height, width, number of batches, number of channels, and strides.
     * - Relies on the `iMatrixAdd` function to perform the actual matrix addition operation.
     *
     * @throws std::invalid_argument When the shapes of `lhs` and `rhs` are not broadcast compatible or their height and width dimensions do not match.
     *
     * @note
     * - The time complexity of this function is O(m * n), where m is the product of the batch and channel dimensions of the output tensor (`out.shape()[0] * out.shape()[1]`), and n is the number of elements in a single matrix (`lhs.shape().H() * lhs.shape().W()`).
     *
     * @code
     * ```cpp
     * // Assume we have a valid tensor type Tensor
     * Tensor out;
     * Tensor lhs;
     * Tensor rhs;
     * try {
     *     tensorMatrixAdd(out, lhs, rhs);
     * } catch (const std::invalid_argument& e) {
     *     std::cerr << e.what() << std::endl;
     * }
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, void>
    tensorMatrixAdd(T& out, const T& lhs, const T& rhs) {
        if (!lhs.shape().isBroadcastCompatible(rhs.shape()) || lhs.shape().H() != rhs.shape().H() || lhs.shape().W() !=
            rhs.shape().
                W()) {
            throw std::invalid_argument("Shapes are not broadcast compatible.");
        }
        std::vector<size_t> offsetC;
        std::vector<size_t> offsetA;
        std::vector<size_t> offsetB;
        const size_t n = lhs.shape().H() * lhs.shape().W();
        for (auto i = 0; i < out.shape()[0]; i++) {
            for (auto j = 0; j < out.shape()[1]; j++) {
                offsetC.push_back(i * out.shape().getStride(0) + j * out.shape().getStride(1));
                offsetA.push_back(i * (lhs.shape().N() > 1 ? lhs.shape().getStride(0) : 0) + j * (lhs.shape().C() > 1
                    ? lhs.shape().getStride(1)
                    : 0));
                offsetB.push_back(i * (rhs.shape().N() > 1 ? rhs.shape().getStride(0) : 0) + j * (
                    rhs.shape().C() > 1 ? rhs.shape().getStride(1) : 0));
            }
        }
        iMatrixAdd(out.data(), lhs.data(), rhs.data(), n, offsetC, offsetA, offsetB);
    }

    DL_API void iMatrixSub(float* out, float* in1, float* in2, size_t n, const std::vector<size_t>& offset_o,
                           const std::vector<size_t>& offset_i1, const std::vector<size_t>& offset_i2);

    /**
     * @brief Performs matrix subtraction operation on tensors with broadcast compatibility.
     *
     * This template function subtracts the tensor `rhs` from the tensor `lhs` and stores the result in the tensor `out`.
     * It is only enabled for types `T` that satisfy `is_valid_tensor_type<T>::value`. The shapes of the input tensors
     * must be broadcast compatible, and their height and width dimensions must match.
     *
     * @tparam T The tensor type, which must meet the condition `is_valid_tensor_type<T>::value`.
     * @param out The output tensor that will hold the result of the subtraction. Memory flow: host-to-function (reference), function-to-host (modified).
     * @param lhs The left-hand side tensor in the subtraction operation. Memory flow: host-to-function.
     * @param rhs The right-hand side tensor in the subtraction operation. Memory flow: host-to-function.
     *
     * @return None
     *
     * **Memory Management Strategy**:
     * - The function does not allocate or free memory for the tensors themselves. It creates local `std::vector` objects (`offsetC`, `offsetA`, `offsetB`) to store offset values. These vectors are automatically managed by their destructors.
     *
     * **Exception Handling Mechanism**:
     * - Throws `std::invalid_argument` if the shapes of `lhs` and `rhs` are not broadcast compatible or if their height and width dimensions do not match.
     *
     * **Relationship with Other Components**:
     * - Depends on the `shape()` method of the tensor type `T` to obtain shape information, such as broadcast compatibility, height, width, batch size, channel count, and strides.
     * - Relies on the `iMatrixSub` function to perform the actual matrix subtraction.
     *
     * @throws std::invalid_argument When the shapes of `lhs` and `rhs` are not broadcast compatible or their height and width dimensions do not match.
     *
     * @note
     * - The time complexity of this function is O(m * n), where m is the product of the batch and channel dimensions of the output tensor (`out.shape()[0] * out.shape()[1]`), and n is the number of elements in a single matrix (`lhs.shape().H() * lhs.shape().W()`).
     *
     * @code
     * ```cpp
     * // Assume we have a valid tensor type Tensor
     * Tensor out;
     * Tensor lhs;
     * Tensor rhs;
     * try {
     *     tensorMatrixSub(out, lhs, rhs);
     * } catch (const std::invalid_argument& e) {
     *     std::cerr << e.what() << std::endl;
     * }
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, void>
    tensorMatrixSub(T& out, const T& lhs, const T& rhs) {
        if (!lhs.shape().isBroadcastCompatible(rhs.shape()) || lhs.shape().H() != rhs.shape().H() || lhs.shape().W() !=
            rhs.shape().
                W()) {
            throw std::invalid_argument("Shapes are not broadcast compatible.");
        }
        std::vector<size_t> offsetC;
        std::vector<size_t> offsetA;
        std::vector<size_t> offsetB;
        const size_t n = lhs.shape().H() * lhs.shape().W();
        for (auto i = 0; i < out.shape()[0]; i++) {
            for (auto j = 0; j < out.shape()[1]; j++) {
                offsetC.push_back(i * out.shape().getStride(0) + j * out.shape().getStride(1));
                offsetA.push_back(i * (lhs.shape().N() > 1 ? lhs.shape().getStride(0) : 0) + j * (lhs.shape().C() > 1
                    ? lhs.shape().getStride(1)
                    : 0));
                offsetB.push_back(i * (rhs.shape().N() > 1 ? rhs.shape().getStride(0) : 0) + j * (
                    rhs.shape().C() > 1 ? rhs.shape().getStride(1) : 0));
            }
        }
        iMatrixSub(out.data(), lhs.data(), rhs.data(), n, offsetC, offsetA, offsetB);
    }

    DL_API void iElementwiseDivide(float* out, float* in1, float* in2, size_t n, const std::vector<size_t>& offset_o,
                                   const std::vector<size_t>& offset_i1, const std::vector<size_t>& offset_i2);

    /**
     * @brief Performs element - wise division operation on tensors with broadcast compatibility.
     *
     * This template function divides each element of the tensor `lhs` by the corresponding element of the tensor `rhs` and stores the result in the tensor `out`.
     * It is only enabled for types `T` that satisfy `is_valid_tensor_type<T>::value`. The shapes of the input tensors must be broadcast compatible, and their height and width dimensions must match.
     *
     * @tparam T The tensor type, which must satisfy `is_valid_tensor_type<T>::value`.
     * @param out The output tensor where the result of the element - wise division will be stored. Memory flow: host - to - function (reference), function - to - host (modified).
     * @param lhs The left - hand side tensor in the division operation. Memory flow: host - to - function.
     * @param rhs The right - hand side tensor in the division operation. Memory flow: host - to - function.
     *
     * @return None
     *
     * **Memory Management Strategy**:
     * - The function does not allocate or free memory for the tensors. It creates local `std::vector` objects (`offsetC`, `offsetA`, `offsetB`) to store offset values. These vectors are automatically managed by their destructors.
     *
     * **Exception Handling Mechanism**:
     * - Throws `std::invalid_argument` if the shapes of `lhs` and `rhs` are not broadcast compatible or if their height and width dimensions do not match.
     *
     * **Relationship with Other Components**:
     * - Depends on the `shape()` method of the tensor type `T` to access shape information, including broadcast compatibility, height, width, batch size, channel count, and strides.
     * - Relies on the `iElementwiseDivide` function to perform the actual element - wise division.
     *
     * @throws std::invalid_argument When the shapes of `lhs` and `rhs` are not broadcast compatible or their height and width dimensions do not match.
     *
     * @note
     * - The time complexity of this function is O(m * n), where m is the product of the batch and channel dimensions of the output tensor (`out.shape()[0] * out.shape()[1]`), and n is the number of elements in a single matrix (`lhs.shape().H() * lhs.shape().W()`).
     *
     * @code
     * ```cpp
     * // Assume we have a valid tensor type Tensor
     * Tensor out;
     * Tensor lhs;
     * Tensor rhs;
     * try {
     *     tensorElementwiseDivide(out, lhs, rhs);
     * } catch (const std::invalid_argument& e) {
     *     std::cerr << e.what() << std::endl;
     * }
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, void>
    tensorElementwiseDivide(T& out, const T& lhs, const T& rhs) {
        if (!lhs.shape().isBroadcastCompatible(rhs.shape()) || lhs.shape().H() != rhs.shape().H() || lhs.shape().W() !=
            rhs.shape().
                W()) {
            throw std::invalid_argument("Shapes are not broadcast compatible.");
        }
        std::vector<size_t> offsetC;
        std::vector<size_t> offsetA;
        std::vector<size_t> offsetB;
        const size_t n = lhs.shape().H() * lhs.shape().W();
        for (auto i = 0; i < out.shape()[0]; i++) {
            for (auto j = 0; j < out.shape()[1]; j++) {
                offsetC.push_back(i * out.shape().getStride(0) + j * out.shape().getStride(1));
                offsetA.push_back(i * (lhs.shape().N() > 1 ? lhs.shape().getStride(0) : 0) + j * (lhs.shape().C() > 1
                    ? lhs.shape().getStride(1)
                    : 0));
                offsetB.push_back(i * (rhs.shape().N() > 1 ? rhs.shape().getStride(0) : 0) + j * (
                    rhs.shape().C() > 1 ? rhs.shape().getStride(1) : 0));
            }
        }
        iElementwiseDivide(out.data(), lhs.data(), rhs.data(), n, offsetC, offsetA, offsetB);
    }

    DL_API void iGeneralMatrixMul(float* A, float* B, float* C, size_t M, size_t N, size_t K,
                                  const std::vector<size_t>& offsetC, const std::vector<size_t>& offsetA,
                                  const std::vector<size_t>& offsetB);

    /**
     * @brief Performs general matrix multiplication on tensors with broadcast compatibility.
     *
     * This template function multiplies the tensor `lhs` by the tensor `rhs` and stores the result in the tensor `out`.
     * It is only enabled for types `T` that satisfy `is_valid_tensor_type<T>::value`. The shapes of the input tensors
     * must be broadcast compatible, and the width of `lhs` must be equal to the height of `rhs`.
     *
     * @tparam T The tensor type, which must satisfy `is_valid_tensor_type<T>::value`.
     * @param out The output tensor that will hold the result of the matrix multiplication. Memory flow: host-to-function (reference), function-to-host (modified).
     * @param lhs The left-hand side tensor in the matrix multiplication. Memory flow: host-to-function.
     * @param rhs The right-hand side tensor in the matrix multiplication. Memory flow: host-to-function.
     *
     * @return None
     *
     * **Memory Management Strategy**:
     * - The function does not allocate or free memory for the tensors themselves. It creates local `std::vector` objects (`offsetC`, `offsetA`, `offsetB`) to store offset values. These vectors are automatically managed by their destructors.
     *
     * **Exception Handling Mechanism**:
     * - Throws `std::invalid_argument` if the shapes of `lhs` and `rhs` are not broadcast compatible or if the width of `lhs` is not equal to the height of `rhs`.
     *
     * **Relationship with Other Components**:
     * - Depends on the `shape()` method of the tensor type `T` to obtain shape information, such as broadcast compatibility, height, width, batch size, channel count, and strides.
     * - Relies on the `iGeneralMatrixMul` function to perform the actual matrix multiplication.
     *
     * @throws std::invalid_argument When the shapes of `lhs` and `rhs` are not broadcast compatible or the width of `lhs` is not equal to the height of `rhs`.
     *
     * @note
     * - The time complexity of this function is O(m * k * n), where m is the height of `lhs`, k is the width of `lhs` (equal to the height of `rhs`), and n is the width of `rhs`.
     *
     * @code
     * ```cpp
     * // Assume we have a valid tensor type Tensor
     * Tensor out;
     * Tensor lhs;
     * Tensor rhs;
     * try {
     *     tensorGeneralMatrixMul(out, lhs, rhs);
     * } catch (const std::invalid_argument& e) {
     *     std::cerr << e.what() << std::endl;
     * }
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, void>
    tensorGeneralMatrixMul(T& out, const T& lhs, const T& rhs) {
        if (!lhs.shape().isBroadcastCompatible(rhs.shape()) || lhs.shape().W() != rhs.shape().H()) {
            throw std::invalid_argument("Shapes are not broadcast compatible.");
        }
        std::vector<size_t> offsetC;
        std::vector<size_t> offsetA;
        std::vector<size_t> offsetB;
        for (auto i = 0; i < out.shape()[0]; i++) {
            for (auto j = 0; j < out.shape()[1]; j++) {
                offsetC.push_back(i * out.shape().getStride(0) + j * out.shape().getStride(1));
                offsetA.push_back(i * (lhs.shape().N() > 1 ? lhs.shape().getStride(0) : 0) + j * (lhs.shape().C() > 1
                    ? lhs.shape().getStride(1)
                    : 0));
                offsetB.push_back(i * (rhs.shape().N() > 1 ? rhs.shape().getStride(0) : 0) + j * (
                    rhs.shape().C() > 1 ? rhs.shape().getStride(1) : 0));
            }
        }
        iGeneralMatrixMul(lhs.data(), rhs.data(), out.data(), lhs.shape().H(), rhs.shape().W(), lhs.shape().W(),
                          offsetC, offsetA, offsetB);
    }

    DL_API void iTensorCoreGEMM(float* A, float* B, float* C, const Dimension& shapeA, const Dimension& shapeB,
                                const Dimension& shapeC);

    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, void>
    GEMMTensorCore(T& out, const T& lhs, const T& rhs) {
        iTensorCoreGEMM(lhs.data(), rhs.data(), out.data(), lhs.shape(), rhs.shape(), out.shape());
    }

    DL_API void iGEMMBackward(float* A, float* B, float* C, const Dimension& shapeA, const Dimension& shapeB,
                              const Dimension& shapeC);

    DL_API void iTranspose(float* out, float* in, size_t rows, size_t cols, const std::vector<size_t>& offset);

    /**
     * @brief Transposes a tensor with a valid tensor type.
     *
     * This template function transposes the input tensor `in` and returns a new tensor `result`. It is only enabled for types `T` that satisfy `is_valid_tensor_type<T>::value`.
     *
     * @tparam T The tensor type, which must satisfy `is_valid_tensor_type<T>::value`.
     * @param in The input tensor to be transposed. Memory flow: host - to - function.
     *
     * @return A new tensor `result` which is the transposed version of the input tensor `in`. Memory flow: function - to - host.
     *
     * **Memory Management Strategy**:
     * - A new tensor `result` is created inside the function to store the transposed data. The memory for this tensor is managed by the tensor type `T` itself.
     * - The function creates a local `std::vector` object `offset` to store offset values. This vector is automatically managed by its destructor.
     *
     * **Exception Handling Mechanism**:
     * - This function does not throw any exceptions explicitly. However, exceptions may be thrown by the constructor of the tensor type `T` or the `iTranspose` function.
     *
     * **Relationship with Other Components**:
     * - Depends on the `shape()` method of the tensor type `T` to access shape information, including dimensions and strides.
     * - Relies on the `iTranspose` function to perform the actual transpose operation.
     *
     * @note
     * - The time complexity of this function is O(m * n), where m is the product of the first two dimensions of the input tensor (`in.shape()[0] * in.shape()[1]`), and n is the product of the last two dimensions (`in.shape()[2] * in.shape()[3]`).
     * - Ensure that the `iTranspose` function is correctly implemented and that the tensor types support the necessary shape and data access methods.
     *
     * @warning
     * - Incorrect implementation of the `iTranspose` function may lead to incorrect results or runtime errors.
     *
     * @code
     * ```cpp
     * // Assume we have a valid tensor type Tensor
     * Tensor in;
     * Tensor transposed = transpose(in);
     * ```
     * @endcode
     */
    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    transpose(const T& in) {
        T result({in.shape()[0], in.shape()[1], in.shape()[3], in.shape()[2]}, in.requiresGrad());
        std::vector<size_t> offset;
        for (auto i = 0; i < in.shape()[0]; i += 1) {
            for (auto j = 0; j < in.shape()[1]; j += 1) {
                offset.push_back(i * in.shape().getStride(0) + j * in.shape().getStride(1));
            }
        }
        iTranspose(result.data(), in.data(), in.shape()[2], in.shape()[3], offset);
        if (in.requiresGrad()) {
            iTranspose(result.grad(), in.grad(), in.shape()[2], in.shape()[3], offset);
        }
        return result;
    }

    DL_API void iSoftmaxJacobian(float* out, float* in, size_t n, const std::vector<size_t>& offset_o,
                                 const std::vector<size_t>& offset_i);

    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    softmaxJacobian(const T& in) {
        const size_t n = std::max(in.shape()[2], in.shape()[3]);
        T result({in.shape()[0], in.shape()[1], n, n});
        std::vector<size_t> offset_o;
        std::vector<size_t> offset_i;
        for (auto i = 0; i < in.shape()[0]; i++) {
            for (auto j = 0; j < in.shape()[1]; j++) {
                offset_o.push_back(i * result.shape().getStride(0) + j * result.shape().getStride(1));
                offset_i.push_back(i * in.shape().getStride(0) + j * in.shape().getStride(1));
            }
        }
        iSoftmaxJacobian(result.data(), in.data(), n, offset_o, offset_i);
        return result;
    }

    DL_API void iImg2col(float* out, float* in, const size_t H_out,
                         const size_t W_out, const size_t C, const size_t K_h, const size_t K_w, const size_t stride,
                         const size_t pad, const size_t H_in, const size_t W_in, const size_t batch);

    DL_API void iImg2colBackward(float* out, float* in, const size_t H_out,
              const size_t W_out, const size_t C, const size_t K_h, const size_t K_w, const size_t stride,
              const size_t pad, const size_t H_in, const size_t W_in, const size_t batch);

    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    tensorImg2col(const T& in, const size_t K_h, const size_t K_w, const size_t stride,
                  const size_t pad) {
        const size_t H_out = (in.shape().H() + 2 * pad - K_h) / stride + 1;
        const size_t W_out = (in.shape().W() + 2 * pad - K_w) / stride + 1;
        T result({in.shape()[0], 1, H_out * W_out, in.shape().C() * K_h * K_w}, in.requiresGrad());
        iImg2col(result.data(), in.data(), H_out, W_out, in.shape().C(), K_h, K_w, stride, pad,
                 in.shape().H(), in.shape().W(), in.shape()[0]);
        if (in.requiresGrad()) {
            iImg2col(result.grad(), in.grad(), H_out, W_out, in.shape().C(), K_h, K_w, stride, pad,
                     in.shape().H(), in.shape().W(), in.shape()[0]);
        }
        return result;
    }

    DL_API void iCol2img(float* out, float* in, size_t H_out,
                         size_t W_out, size_t C_out, size_t batches);

    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    tensorCol2img(const T& in, const size_t H_out, const size_t W_out) {
        T result({in.shape()[0], in.shape()[3], H_out, W_out}, in.requiresGrad());
        iCol2img(result.data(), in.data(), H_out, W_out, in.shape()[3], in.shape()[0]);
        if (in.requiresGrad()) {
            iCol2img(result.grad(), in.grad(), H_out, W_out, in.shape()[3], in.shape()[0]);
        }
        return result;
    }

    DL_API void iCol2imgBackward(float* out, float* in, size_t H_out, size_t W_out, size_t C_out, size_t batches);

    DL_API void iAveragePooling(float* out, float* in,
                                size_t pool_size, size_t stride, size_t padding,
                                size_t batches, size_t channels, size_t H_in, size_t W_in,
                                size_t H_out, size_t W_out);

    template <typename T>
    std::enable_if_t<is_valid_tensor_type<T>::value, T>
    tensorAveragePooling(const T& in, const size_t pool_size, const size_t stride,
                          const size_t padding) {
        const size_t H_out = OUTPUT_DIM(in.shape().H(), pool_size, stride, padding);
        const size_t W_out = OUTPUT_DIM(in.shape().W(), pool_size, stride, padding);
        T result({in.shape()[0], in.shape()[1], H_out, W_out}, in.requiresGrad());
        iAveragePooling(result.data(), in.data(), pool_size, stride, padding,
                        in.shape()[0], in.shape()[1], in.shape().H(), in.shape().W(),
                        H_out, W_out);
        if (in.requiresGrad()) {
            iAveragePooling(result.grad(), in.grad(), pool_size, stride, padding,
                            in.shape()[0], in.shape()[1], in.shape().H(), in.shape().W(),
                            H_out, W_out);
        }
        return result;
    }

    DL_API void iAveragePoolingBackward(float* out, float* in,
                                size_t pool_size, size_t stride, size_t padding,
                                size_t batches, size_t channels, size_t H_in, size_t W_in,
                                size_t H_out, size_t W_out);
}
#endif //TENSOROPERATIONS_CUH

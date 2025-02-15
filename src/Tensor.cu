#include "NeuZephyr/Tensor.cuh"
#include "NeuZephyr/utils.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include <curand.h>

namespace nz::data {
    /**
     * @brief Overloads the `<<` operator to print the tensor's data to an output stream.
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the output stream operator (`<<`) to print the contents of a tensor to the specified
     * output stream (e.g., `std::cout` or a file stream).
     *
     * The tensor's data is first copied from GPU memory to host memory for printing, and then the data
     * is printed in a 2D matrix format. Each row of the tensor is printed on a new line, and each element
     * in a row is separated by a space. Each row is enclosed in square brackets.
     *
     * @param os The output stream to which the tensor will be printed.
     * @param tensor The tensor whose contents will be printed.
     * @return The output stream (`os`) after the tensor has been printed, allowing for chaining of operations.
     *
     * @note
     * - This operator works by accessing the tensor's private data members (e.g., `_data`) directly.
     * - The tensor's data is assumed to be in a valid state (i.e., properly allocated in GPU memory) before printing.
     * - The function copies the tensor's data from device (GPU) memory to host (CPU) memory using `cudaMemcpy`, which
     *   may introduce performance overhead for large tensors.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(1.0f);  // Fill the tensor with 1.0f
     * std::cout << tensor << std::endl;  // Prints the tensor to standard output in matrix format
     * ```
     * @endcode
     */
    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        auto* data = static_cast<Tensor::value_type*>(malloc(tensor._size * sizeof(Tensor::value_type)));
        cudaMemcpy(data, tensor._data, tensor._size * sizeof(Tensor::value_type), cudaMemcpyDeviceToHost);
        std::ostream_iterator<Tensor::value_type> output_iterator(os, " ");
        for (int i = 0; i < tensor._shape[0]; ++i) {
            const auto it = data + i * tensor._shape[1];
            const auto it_end = it + tensor._shape[1];
            os << "[";
            std::copy(it, it_end, output_iterator);
            os << "]";
            os << std::endl;
        }
        if (tensor._requires_grad) {
            os << "Gradient: " << std::endl;
            auto* grad = static_cast<Tensor::value_type*>(malloc(tensor._size * sizeof(Tensor::value_type)));
            cudaMemcpy(grad, tensor._grad, tensor._size * sizeof(Tensor::value_type), cudaMemcpyDeviceToHost);
            for (int i = 0; i < tensor._shape[0]; ++i) {
                const auto it = grad + i * tensor._shape[1];
                const auto it_end = it + tensor._shape[1];
                os << "[";
                std::copy(it, it_end, output_iterator);
                os << "]";
                os << std::endl;
            }
            free(grad);
        }
        free(data);
        return os;
    }

    /**
     * @brief Overloads the `>>` operator to read a tensor's data from an input stream.
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the input stream operator (`>>`) to read the contents of a tensor from the specified
     * input stream (e.g., `std::cin` or a file stream).
     *
     * The function reads the tensor's data element by element from the input stream and stores
     * the values in a temporary buffer. Once all the data has been read, it is copied from the
     * host memory back into the tensor's GPU memory using `cudaMemcpy`.
     *
     * @param is The input stream from which the tensor's data will be read.
     * @param tensor The tensor to which the data will be read.
     * @return The input stream (`is`) after reading the tensor's data, allowing for chaining of operations.
     *
     * @note
     * - This operator works by reading data from the input stream and storing it in a temporary buffer on the host.
     * - The function assumes that the input data matches the size of the tensor. If the data is malformed or does not
     *   match, the behavior may be undefined.
     * - After reading, the data is copied from host memory back into the tensor's GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * std::cin >> tensor;  // Reads the tensor's data from standard input
     * ```
     * @endcode
     */
    std::istream& operator>>(std::istream& is, const Tensor& tensor) {
        auto* data = static_cast<Tensor::value_type*>(malloc(tensor._size * sizeof(Tensor::value_type)));
        for (int i = 0; i < tensor._size; ++i) {
            is >> data[i];
        }
        cudaMemcpy(tensor._data, data, tensor._size * sizeof(Tensor::value_type), cudaMemcpyHostToDevice);
        free(data);
        return is;
    }

    /**
     * @brief Multiplies a tensor by a scalar (element-wise multiplication).
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the multiplication operator (`*`) to multiply each element of the tensor by a scalar value.
     * It performs element-wise multiplication, where every element in the tensor is multiplied
     * by the given scalar.
     *
     * @param lhs The scalar value to multiply each element of the tensor by.
     * @param rhs The tensor whose elements will be multiplied by the scalar.
     * @return A new tensor containing the result of the element-wise multiplication.
     *
     * This function uses a CUDA kernel (`ScalarMul`) to perform the element-wise multiplication in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise multiplication.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(1.0f);  // Fill the tensor with 1.0f
     * float scalar = 2.0f;
     * Tensor result = scalar * tensor;  // Multiply each element of the tensor by 2.0f
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor operator*(const Tensor::value_type lhs, const Tensor& rhs) {
        Tensor result(rhs._shape, rhs._requires_grad);
        dim3 block(256);
        dim3 grid((rhs._size + block.x - 1) / block.x);
        krnl::ScalarMul(grid, block, result._data, rhs._data, lhs, rhs._size);
        return result;
    }

    /**
     * @brief Multiplies a tensor by a scalar (element-wise multiplication).
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the multiplication operator (`*`) to multiply each element of the tensor by a scalar value.
     * It performs element-wise multiplication, where every element in the tensor is multiplied
     * by the given scalar.
     *
     * @param lhs The tensor whose elements will be multiplied by the scalar.
     * @param rhs The scalar value to multiply each element of the tensor by.
     * @return A new tensor containing the result of the element-wise multiplication.
     *
     * This function uses a CUDA kernel (`ScalarMul`) to perform the element-wise multiplication in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise multiplication.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(1.0f);  // Fill the tensor with 1.0f
     * float scalar = 2.0f;
     * Tensor result = tensor * scalar;  // Multiply each element of the tensor by 2.0f
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor operator*(const Tensor& lhs, const Tensor::value_type rhs) {
        Tensor result(lhs._shape, lhs._requires_grad);
        dim3 block(256);
        dim3 grid((lhs._size + block.x - 1) / block.x);
        krnl::ScalarMul(grid, block, result._data, lhs._data, rhs, lhs._size);
        return result;
    }

    /**
     * @brief Divides a tensor by a scalar (element-wise division).
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the division operator (`/`) to divide each element of the tensor by a scalar value.
     * It performs element-wise division, where every element in the tensor is divided by the given scalar.
     *
     * @param lhs The tensor whose elements will be divided by the scalar.
     * @param rhs The scalar value by which each element of the tensor will be divided.
     * @return A new tensor containing the result of the element-wise division.
     *
     * This function uses a CUDA kernel (`ScalarDiv`) to perform the element-wise division in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise division.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     * - Division by zero should be handled appropriately, and input tensors should be checked to ensure no element is zero.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(10.0f);  // Fill the tensor with 10.0f
     * float scalar = 2.0f;
     * Tensor result = tensor / scalar;  // Divide each element of the tensor by 2.0f
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor operator/(const Tensor& lhs, const Tensor::value_type rhs) {
        Tensor result(lhs._shape, lhs._requires_grad);
        dim3 block(256);
        dim3 grid((lhs._size + block.x - 1) / block.x);
        krnl::ScalarDiv(grid, block, result._data, lhs._data, rhs, lhs._size);
        return result;
    }

    /**
     * @brief Adds a scalar to a tensor (element-wise addition).
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the addition operator (`+`) to add a scalar value to each element of the tensor.
     * It performs element-wise addition, where every element in the tensor is increased by the given scalar.
     *
     * @param lhs The tensor whose elements will be added by the scalar.
     * @param rhs The scalar value to add to each element of the tensor.
     * @return A new tensor containing the result of the element-wise addition.
     *
     * This function uses a CUDA kernel (`ScalarAdd`) to perform the element-wise addition in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise addition.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(1.0f);  // Fill the tensor with 1.0f
     * float scalar = 2.0f;
     * Tensor result = tensor + scalar;  // Add 2.0f to each element of the tensor
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor operator+(const Tensor& lhs, const Tensor::value_type rhs) {
        Tensor result(lhs._shape, lhs._requires_grad);
        dim3 block(256);
        dim3 grid((lhs._size + block.x - 1) / block.x);
        krnl::ScalarAdd(grid, block, result._data, lhs._data, rhs, lhs._size);
        return result;
    }

    /**
     * @brief Adds a scalar to a tensor (element-wise addition).
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the addition operator (`+`) to add a scalar value to each element of the tensor.
     * It performs element-wise addition, where every element in the tensor is increased by the given scalar.
     *
     * @param lhs The scalar value to add to each element of the tensor.
     * @param rhs The tensor whose elements will be added by the scalar.
     * @return A new tensor containing the result of the element-wise addition.
     *
     * This function uses a CUDA kernel (`ScalarAdd`) to perform the element-wise addition in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise addition.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(1.0f);  // Fill the tensor with 1.0f
     * float scalar = 2.0f;
     * Tensor result = scalar + tensor;  // Add 2.0f to each element of the tensor
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor operator+(const Tensor::value_type lhs, const Tensor& rhs) {
        Tensor result(rhs._shape, rhs._requires_grad);
        dim3 block(256);
        dim3 grid((rhs._size + block.x - 1) / block.x);
        krnl::ScalarAdd(grid, block, result._data, rhs._data, lhs, rhs._size);
        return result;
    }

    /**
     * @brief Subtracts a scalar from a tensor (element-wise subtraction).
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the subtraction operator (`-`) to subtract a scalar value from each element of the tensor.
     * It performs element-wise subtraction, where every element in the tensor is decreased by the given scalar.
     *
     * @param lhs The tensor whose elements will have the scalar subtracted from them.
     * @param rhs The scalar value to subtract from each element of the tensor.
     * @return A new tensor containing the result of the element-wise subtraction.
     *
     * This function uses a CUDA kernel (`ScalarAdd`) to perform the element-wise subtraction in parallel
     * on the GPU. The result is stored in a new tensor, which is returned. The scalar is negated during the
     * operation to achieve subtraction.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise subtraction.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(10.0f);  // Fill the tensor with 10.0f
     * float scalar = 2.0f;
     * Tensor result = tensor - scalar;  // Subtract 2.0f from each element of the tensor
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor operator-(const Tensor& lhs, const Tensor::value_type rhs) {
        Tensor result(lhs._shape, lhs._requires_grad);
        dim3 block(256);
        dim3 grid((lhs._size + block.x - 1) / block.x);
        krnl::ScalarAdd(grid, block, result._data, lhs._data, -rhs, lhs._size);
        return result;
    }

    /**
     * @brief Subtracts a tensor from a scalar (element-wise subtraction).
     *
     * This function is a friend of the `Tensor` class and provides an overloaded version of
     * the subtraction operator (`-`) to subtract each element of the tensor from a scalar value.
     * It performs element-wise subtraction, where every element in the tensor is subtracted from the given scalar.
     *
     * @param lhs The scalar value from which each element of the tensor will be subtracted.
     * @param rhs The tensor whose elements will be subtracted from the scalar.
     * @return A new tensor containing the result of the element-wise subtraction.
     *
     * This function uses a CUDA kernel (`ScalarAdd`) to perform the element-wise subtraction in parallel
     * on the GPU. The result is stored in a new tensor, which is returned. The scalar is negated during the
     * operation to achieve subtraction.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise subtraction.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     * - The scalar is negated to perform subtraction, which results in `lhs - rhs` for each element in the tensor.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(10.0f);  // Fill the tensor with 10.0f
     * float scalar = 2.0f;
     * Tensor result = scalar - tensor;  // Subtract each element of the tensor from 2.0f
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor operator-(const Tensor::value_type lhs, const Tensor& rhs) {
        Tensor result(rhs._shape, rhs._requires_grad);
        dim3 block(256);
        dim3 grid((rhs._size + block.x - 1) / block.x);
        krnl::ScalarAdd(grid, block, result._data, rhs._data, -lhs, rhs._size);
        return result;
    }

    /**
     * @brief Applies the Rectified Linear Unit (ReLU) activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the ReLU activation function element-wise
     * to the given tensor. ReLU is a popular activation function in neural networks that replaces all negative
     * values in the tensor with zero, leaving positive values unchanged.
     *
     * @param tensor The tensor to which the ReLU activation function will be applied.
     * @return A new tensor containing the result of the ReLU activation applied element-wise to the input tensor.
     *
     * This function uses a CUDA kernel (`RectifiedLinearUnit`) to perform the element-wise activation in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise ReLU activation.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(-1.0f);  // Fill the tensor with negative values
     * Tensor result = ReLU(tensor);  // Apply the ReLU activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor ReLU(const Tensor& tensor) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        krnl::RectifiedLinearUnit(grid, block, result._data, tensor._data, tensor._size);
        return result;
    }

    /**
     * @brief Applies the Sigmoid activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the Sigmoid activation function element-wise
     * to the given tensor. The Sigmoid function squashes each element of the tensor to a value between 0 and 1.
     * The Sigmoid function is commonly used in neural networks, especially for binary classification tasks.
     *
     * @param tensor The tensor to which the Sigmoid activation function will be applied.
     * @return A new tensor containing the result of the Sigmoid activation applied element-wise to the input tensor.
     *
     * This function uses a CUDA kernel (`Sigmoid`) to perform the element-wise activation in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise Sigmoid activation.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(0.5f);  // Fill the tensor with values between 0 and 1
     * Tensor result = Sigmoid(tensor);  // Apply the Sigmoid activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor Sigmoid(const Tensor& tensor) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        krnl::Sigmoid(grid, block, result._data, tensor._data, tensor._size);
        return result;
    }

    /**
     * @brief Applies the Tanh (hyperbolic tangent) activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the Tanh activation function element-wise
     * to the given tensor. The Tanh function squashes each element of the tensor to a value between -1 and 1.
     * The Tanh function is often used in neural networks and is similar to the Sigmoid function but has a wider output range.
     *
     * @param tensor The tensor to which the Tanh activation function will be applied.
     * @return A new tensor containing the result of the Tanh activation applied element-wise to the input tensor.
     *
     * This function uses a CUDA kernel (`Tanh`) to perform the element-wise activation in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise Tanh activation.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(0.5f);  // Fill the tensor with values
     * Tensor result = Tanh(tensor);  // Apply the Tanh activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor Tanh(const Tensor& tensor) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        krnl::Tanh(grid, block, result._data, tensor._data, tensor._size);
        return result;
    }

    /**
     * @brief Applies the Tanh (hyperbolic tangent) activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the Tanh activation function element-wise
     * to the given tensor. The Tanh function squashes each element of the tensor to a value between -1 and 1.
     * The Tanh function is often used in neural networks and is similar to the Sigmoid function but has a wider output range.
     *
     * @param tensor The tensor to which the Tanh activation function will be applied.
     * @return A new tensor containing the result of the Tanh activation applied element-wise to the input tensor.
     *
     * This function uses a CUDA kernel (`Tanh`) to perform the element-wise activation in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise Tanh activation.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(0.5f);  // Fill the tensor with values
     * Tensor result = Tanh(tensor);  // Apply the Tanh activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor LeakyReLU(const Tensor& tensor, float alpha) {
        Tensor result(tensor._shape, tensor._requires_grad);
        const dim3 block(256);
        const dim3 grid((tensor._size + block.x - 1) / block.x);
        krnl::LeakyReLU(grid, block, result._data, tensor._data, tensor._size, alpha);
        return result;
    }

    /**
     * @brief Applies the Swish activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the Swish activation function element-wise
     * to the given tensor. Swish is a smooth, non-monotonic activation function defined as:
     *
     *     Swish(x) = x / (1 + exp(-x))
     *
     * Swish has been shown to work well in deep neural networks, offering benefits over ReLU in certain tasks.
     *
     * @param tensor The tensor to which the Swish activation function will be applied.
     * @return A new tensor containing the result of the Swish activation applied element-wise to the input tensor.
     *
     * This function uses a CUDA kernel (`Swish`) to perform the element-wise activation in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise Swish activation.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(0.5f);  // Fill the tensor with values
     * Tensor result = Swish(tensor);  // Apply the Swish activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor Swish(const Tensor& tensor) {
        Tensor result(tensor._shape, tensor._requires_grad);
        const dim3 block(256);
        const dim3 grid((tensor._size + block.x - 1) / block.x);
        krnl::Swish(grid, block, result._data, tensor._data, tensor._size);
        return result;
    }

    /**
     * @brief Applies the Exponential Linear Unit (ELU) activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the ELU activation function element-wise
     * to the given tensor. ELU is a smooth, differentiable activation function defined as:
     *
     *     ELU(x) = x, if x > 0
     *     ELU(x) = alpha * (exp(x) - 1), if x <= 0
     *
     * where `alpha` is a hyperparameter that controls the slope of the negative part of the function.
     * ELU has been shown to perform better than ReLU in certain situations, especially in deep networks.
     *
     * @param tensor The tensor to which the ELU activation function will be applied.
     * @param alpha A parameter that controls the value for negative inputs. It determines the slope of the negative part.
     * @return A new tensor containing the result of the ELU activation applied element-wise to the input tensor.
     *
     * This function uses a CUDA kernel (`ExponentialLinearUnit`) to perform the element-wise activation in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise ELU activation.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(-1.0f);  // Fill the tensor with negative values
     * float alpha = 1.0f;
     * Tensor result = ELU(tensor, alpha);  // Apply the ELU activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor ELU(const Tensor& tensor, float alpha) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        krnl::ExponentialLinearUnit(grid, block, result._data, tensor._data, tensor._size, alpha);
        return result;
    }

    /**
     * @brief Applies the Hard Sigmoid activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the Hard Sigmoid activation function element-wise
     * to the given tensor. Hard Sigmoid is a piecewise linear approximation of the Sigmoid function, defined as:
     *
     *     HardSigmoid(x) = alpha * x + beta, if x is between -2.5 and 2.5
     *     HardSigmoid(x) = 0, if x < -2.5
     *     HardSigmoid(x) = 1, if x > 2.5
     *
     * where `alpha` and `beta` are parameters that control the slope and the shift of the function.
     * Hard Sigmoid is often used in neural networks as a computationally efficient approximation to Sigmoid.
     *
     * @param tensor The tensor to which the Hard Sigmoid activation function will be applied.
     * @param alpha A parameter that controls the slope of the activation function.
     * @param beta A parameter that controls the shift of the activation function.
     * @return A new tensor containing the result of the Hard Sigmoid activation applied element-wise to the input tensor.
     *
     * This function uses a CUDA kernel (`HardSigmoid`) to perform the element-wise activation in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise Hard Sigmoid activation.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(1.0f);  // Fill the tensor with values
     * float alpha = 0.2f;
     * float beta = 0.5f;
     * Tensor result = HardSigmoid(tensor, alpha, beta);  // Apply the Hard Sigmoid activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor HardSigmoid(const Tensor& tensor, float alpha, float beta) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        krnl::HardSigmoid(grid, block, result._data, tensor._data, tensor._size, alpha, beta);
        return result;
    }

    /**
     * @brief Applies the Hard Swish activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the Hard Swish activation function element-wise
     * to the given tensor. Hard Swish is a variant of the Swish activation function, which is defined as:
     *
     *     HardSwish(x) = x * (ReLU6(x + 3) / 6)
     *
     * where `ReLU6(x)` is the Rectified Linear Unit function with a maximum value of 6. The Hard Swish function
     * is computationally more efficient than the original Swish function while maintaining similar properties.
     *
     * @param tensor The tensor to which the Hard Swish activation function will be applied.
     * @param alpha A parameter that controls the scaling of the activation. Typically, this is set to 1.0.
     * @param beta A parameter that controls the shift of the activation. Typically, this is set to 3.0.
     * @return A new tensor containing the result of the Hard Swish activation applied element-wise to the input tensor.
     *
     * This function uses a CUDA kernel (`HardSwish`) to perform the element-wise activation in parallel
     * on the GPU. The result is stored in a new tensor, which is returned.
     *
     * @note
     * - This operator does not modify the original tensor. Instead, it returns a new tensor that contains the
     *   result of the element-wise Hard Swish activation.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(1.0f);  // Fill the tensor with values
     * float alpha = 1.0f;
     * float beta = 3.0f;
     * Tensor result = HardSwish(tensor, alpha, beta);  // Apply the Hard Swish activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor HardSwish(const Tensor& tensor, float alpha, float beta) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        krnl::HardSwish(grid, block, result._data, tensor._data, tensor._size, alpha, beta);
        return result;
    }

    /**
     * @brief Applies the Softmax activation function to a tensor.
     *
     * This function is a friend of the `Tensor` class and applies the Softmax activation function element-wise
     * to the given tensor. The Softmax function converts the tensor into a probability distribution, where each
     * element is transformed into a value between 0 and 1, and the sum of all elements in the tensor equals 1.
     * The Softmax function is commonly used in the output layer of neural networks for multi-class classification tasks.
     *
     * The Softmax function for each element `x_i` in the tensor is computed as:
     *
     *     Softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
     *
     * @param tensor The tensor to which the Softmax activation function will be applied.
     * @return The input tensor with the Softmax function applied element-wise.
     *
     * This function uses a CUDA kernel (`SummationExp`) to compute the sum of the exponentiated values in parallel
     * on the GPU, and then another kernel (`Softmax`) to apply the Softmax transformation. The result is stored in
     * the original tensor, which is returned.
     *
     * @note
     * - This operator modifies the original tensor by applying the Softmax transformation in-place.
     * - The function assumes that the tensor is already in a valid state and that the tensor's data is in GPU memory.
     * - The Softmax computation is performed in two stages: first by calculating the exponentiated values' sum,
     *   then applying the Softmax transformation.
     *
     * @code
     * ```cpp
     * Tensor tensor({2, 3});
     * tensor.fill(1.0f);  // Fill the tensor with values
     * Tensor result = Softmax(tensor);  // Apply the Softmax activation
     * std::cout << result << std::endl;  // Print the resulting tensor
     * ```
     * @endcode
     */
    Tensor Softmax(const Tensor& tensor) {
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        float* result_d;
        float* result_h;
        float sum = 0;
        cudaMalloc(&result_d, grid.x * sizeof(Tensor::value_type));
        result_h = (float*)malloc(grid.x * sizeof(Tensor::value_type));
        krnl::SummationExp(grid, block, block.x * sizeof(float), result_d, tensor._data, tensor._size);
        cudaMemcpy(result_h, result_d, grid.x * sizeof(Tensor::value_type), cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid.x; i++) {
            sum += result_h[i];
        }
        free(result_h);
        cudaFree(result_d);
        krnl::Softmax(grid, block, tensor._data, tensor._data, sum, tensor._size);
        return tensor;
    }


    // Constructors
    Tensor::Tensor() :
        _size(0), _shape({0, 0}), _data(nullptr), _grad(nullptr), _requires_grad(false) {
    }

    Tensor::Tensor(const shape_type& shape, const bool requires_grad) // NOLINT(*-pro-type-member-init)
        :
        _size(shape[0] * shape[1]), _shape(shape), _requires_grad(requires_grad) {
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        }
        else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const shape_type& shape, const value_type* data, const bool requires_grad, const bool host) :
        _size(shape[0] * shape[1]), _shape(shape), _requires_grad(requires_grad) {
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        if (host) {
            cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        }
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        }
        else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const shape_type& shape, const std::initializer_list<value_type>& data, const bool requires_grad)  :
        _size(shape[0] * shape[1]), _shape(shape), _requires_grad(requires_grad) {
        if (std::distance(data.begin(), data.end()) < _size) {
            throw std::invalid_argument("Initializer list size is less than the tensor size.");
        }
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        } else {
            _grad = nullptr;
        }
        auto host_buf = new value_type[_size];
        auto it = data.begin();
        for (auto i = 0; i < _size; ++i, ++it) {
            host_buf[i] = *it;
        }
        cudaMemcpy(_data, host_buf, _size * sizeof(value_type), cudaMemcpyHostToDevice);
        delete[] host_buf;
    }

    // Copy and Move constructors
    Tensor::Tensor(const Tensor& other) :
        _size(other._size), _shape(other._shape), _requires_grad(other._requires_grad) {
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemcpy(_data, other._data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
            cudaMemcpy(_grad, other._grad, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        }
        else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(Tensor&& other) noexcept :
        _size(other._size), _shape(std::move(other._shape)), _requires_grad(other._requires_grad) {
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        cudaMemcpy(_data, other._data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMemcpy(_grad, other._grad, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        }
        other._data = nullptr;
        other._grad = nullptr;
    }

    // Assignment operators
    Tensor& Tensor::operator=(const Tensor& other) {
        if (this != &other) {
            _size = other._size;
            _shape = other._shape;
            _requires_grad = other._requires_grad;
            cudaFree(_data);
            cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
            cudaMemcpy(_data, other._data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
            if (_requires_grad) {
                cudaFree(_grad);
                cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
                cudaMemcpy(_grad, other._grad, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            _size = other._size;
            _shape = std::move(other._shape);
            cudaMemcpy(_data, other._data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
            if (_requires_grad) {
                cudaMemcpy(_grad, other._grad, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
            }
            other._data = nullptr;
            other._grad = nullptr;
        }
        return *this;
    }

    // Destructor
    Tensor::~Tensor() {
        cudaFree(_data);
        if (_requires_grad) {
            cudaFree(_grad);
        }
    }

    // Getter methods
    bool Tensor::requiresGrad() const noexcept { return _requires_grad; }
    Tensor::shape_type Tensor::shape() const noexcept { return _shape; }
    Tensor::size_type Tensor::size() const noexcept { return _size; }

    // Setter methods
    void Tensor::setRequiresGrad(const bool requires_grad) noexcept {
        if (requires_grad && _grad == nullptr) {
            cudaMalloc(reinterpret_cast<value_type**>(_grad), _size * sizeof(value_type));
        }
        if (!requires_grad && _grad != nullptr) {
            cudaFree(_grad);
            _grad = nullptr;
        }
        _requires_grad = requires_grad;
    }

    void Tensor::dataInject(const std::initializer_list<value_type>& data, const bool grad) const {
        dataInject(data.begin(), data.end(), grad);
    }

    // Operations
    void Tensor::zeroGrad() const noexcept {
        if (_requires_grad) {
            cudaMemset(_grad, 0, _size * sizeof(value_type));
        }
    }

    void Tensor::print() const noexcept {
        const std::ostream_iterator<value_type> output_iterator(std::cout, " ");
        auto* data = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        cudaMemcpy(data, _data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        for (size_type i = 0; i < _shape[0]; ++i) {
            const auto it = data + i * _shape[1];
            const auto end_it = it + _shape[1];
            std::cout << "[";
            std::copy(it, end_it, output_iterator);
            std::cout << "]";
            std::cout << std::endl;
        }
        free(data);
    }

    void Tensor::dataInject(const value_type* data, const bool grad) const {
        if (grad) {
            if (_requires_grad) {
                if (const cudaError_t error = cudaMemcpy(_grad, data, _size * sizeof(value_type), cudaMemcpyHostToDevice); error != cudaSuccess) {
                    throw std::runtime_error("Failed to copy gradient data to device");
                }
            } else {
                throw std::runtime_error("Tensor does not require gradients");
            }
        } else {
            if (const cudaError_t error = cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyHostToDevice); error != cudaSuccess) {
                throw std::runtime_error("Failed to copy data to device");
            }
        }
    }

    void Tensor::randomize(unsigned long long seed) const {
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandGenerateUniform(gen, _data, _size);
    }

    void Tensor::clear() const {
        cudaMemset(_data, 0, _size * sizeof(value_type));
    }

    void Tensor::fill(const value_type value) const {
        auto* data = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        for (size_type i = 0; i < _size; ++i) {
            data[i] = value;
        }
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyHostToDevice);
        free(data);
    }

    void Tensor::fillGrad(const value_type value) const {
        auto* grad = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        for (size_type i = 0; i < _size; ++i) {
            grad[i] = value;
        }
        cudaMemcpy(_grad, grad, _size * sizeof(value_type), cudaMemcpyHostToDevice);
        free(grad);
    }

    Tensor Tensor::operator+(const Tensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor result(_shape, _requires_grad);
        const dim3 block(256);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::MatrixAdd(grid, block, _data, other._data, result._data, _size);
        return result;
    }

    Tensor Tensor::operator-(const Tensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor result(_shape, _requires_grad);
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        krnl::MatrixSub(grid, block, _data, other._data, result._data, _size);
        return result;
    }

    Tensor Tensor::operator*(const Tensor& other) const {
        if (_shape[1] != other._shape[0]) {
            throw std::invalid_argument("Matrix shapes do not match");
        }
        Tensor result({_shape[0], other._shape[1]}, _requires_grad);
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((result._shape[1] + block.x - 1) / block.x, (result._shape[0] + block.y - 1) / block.y);
        krnl::GeneralMatrixMul(grid, block, _data, other._data, result._data, _shape[0], other._shape[1],
                                                   _shape[1]);
        return result;
    }

    void Tensor::reshape(const shape_type& shape) {
        if (shape[0] * shape[1] != _size) {
            WARN("Reshaping to a different size will cause data loss");
        }
        auto* temp = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        cudaMemcpy(temp, _data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        cudaFree(_data);
        value_type* temp_grad = nullptr;
        if (_requires_grad) {
            temp_grad = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
            cudaMemcpy(temp_grad, _grad, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
            cudaFree(_grad);
        }
        const size_type size = _size;
        _size = shape[0] * shape[1];
        _shape = shape;
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemset(_data, 0, _size * sizeof(value_type));
        cudaMemcpy(_data, temp, size * sizeof(value_type), cudaMemcpyHostToDevice);
        free(temp);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
            cudaMemset(_grad, 0, _size * sizeof(value_type));
            cudaMemcpy(_grad, temp_grad, size * sizeof(value_type), cudaMemcpyHostToDevice);
            free(temp_grad);
        }
    }

    void Tensor::reshape(const std::initializer_list<int>& shape) {
        reshape(shape_type(shape));
    }

    void Tensor::transpose() {
        value_type* temp;
        cudaMalloc((value_type**)&temp, _size * sizeof(value_type));
        cudaMemcpy(temp, _data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((_shape[0] + block.x - 1) / block.x, (_shape[1] + block.y - 1) / block.y);
        krnl::Transpose(grid, block, temp, _data, _shape[0], _shape[1]);
        reshape({_shape[1], _shape[0]});
        cudaFree(temp);
    }

    void Tensor::setData(const shape_type& position, const value_type value) const {
        if (position[0] >= _shape[0] || position[1] >= _shape[1]) {
            throw std::invalid_argument("Invalid position");
        }
        auto* data = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        cudaMemcpy(data, _data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        data[position[0] * _shape[1] + position[1]] = value;
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyHostToDevice);
        free(data);
    }

    void Tensor::setData(const std::initializer_list<int>& position, const value_type value) const {
        setData(shape_type(position), value);
    }

    Tensor::value_type* Tensor::data() const noexcept {
        return _data;
    }

    Tensor::value_type* Tensor::grad() const {
        if (!_requires_grad) {
            throw std::runtime_error("Tensor does not require gradients");
        }
        return _grad;
    }

    std::ostream& Tensor::printGrad(std::ostream& os) const {
        auto* data = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        cudaMemcpy(data, _grad, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        std::ostream_iterator<value_type> output_iterator(os, " ");
        for (int i = 0; i < _shape[0]; ++i) {
            const auto it = data + i * _shape[1];
            const auto it_end = it + _shape[1];
            os << "[";
            std::copy(it, it_end, output_iterator);
            os << "]";
            os << std::endl;
        }
        free(data);
        return os;
    }

    Tensor Tensor::operator-() const {
        Tensor result(_shape, _requires_grad);
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        krnl::Negation(grid, block, result._data, _data, _size);
        return result;
    }

    void Tensor::recip() const {
        value_type* data;
        cudaMalloc(reinterpret_cast<value_type**>(&data), _size * sizeof(value_type));
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        krnl::Recip(grid, block, data, _data, _size);
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        cudaFree(data);
    }


}

#include "NeuZephyr/MappedTensor.cuh"

#include <chrono>
#include <iterator>
#include <iostream>
#include <curand.h>

#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/utils.cuh"
#include "NeuZephyr/StreamManager.cuh"

namespace nz::data {
    /**
     * @brief Overload the << operator to print a MappedTensor object to an output stream.
     *
     * @param os An output stream (host-to-host) where the MappedTensor data and gradient will be printed.
     * @param tensor A constant reference (host-to-host) to the MappedTensor object to be printed.
     *
     * @return A reference to the output stream `os` after printing the tensor data and possibly its gradient.
     *
     * This function provides a convenient way to print a MappedTensor object using the << operator. It first calls the `print` method of the MappedTensor to print the tensor's data. If the tensor requires gradients, it then prints a header "Gradient: " followed by the gradient data using the `printGrad` method.
     *
     * Memory management: The function does not allocate or deallocate any memory. It relies on the `print` and `printGrad` methods of the MappedTensor, which also do not perform memory allocation.
     * Exception handling: If the tensor requires gradients and an exception occurs during the `printGrad` call (e.g., due to an invalid state of the output stream or incorrect internal data), the exception will be propagated. If the tensor does not require gradients, the `printGrad` call is skipped, and no exception related to gradient printing will be thrown.
     * Relationship with other components: This function is related to the data presentation component of the MappedTensor. It integrates the `print` and `printGrad` methods to provide a unified way of printing the tensor and its gradient.
     *
     * @throws std::invalid_argument Propagated from the `printGrad` method if the tensor requires gradients and there is an issue with gradient printing.
     *
     * @note
     * - The overall time complexity of this function is O(m * n) if the tensor does not require gradients and O(2 * m * n) if it does, where m is the number of rows (`_shape[0]`) and n is the number of columns (`_shape[1]`) of the tensor, as it iterates over the tensor data and possibly the gradient data.
     * - Ensure that the output stream `os` is in a valid state before calling this function.
     *
     * @code
     * ```cpp
     * nz::data::MappedTensor::shape_type shape = {2, 3};
     * nz::data::MappedTensor tensor(shape, true);
     * tensor.dataInject({1, 2, 3, 4, 5, 6}, false);
     * tensor.dataInject({7, 8, 9, 10, 11, 12}, true);
     * std::cout << tensor;
     * ```
     * @endcode
     */
    std::ostream& operator<<(std::ostream& os, const MappedTensor& tensor) {
        tensor.print(os);
        if (tensor._requires_grad) {
            os << "Gradient: " << std::endl;
            tensor.printGrad(os);
        }
        return os;
    }

    /**
     * @brief Overload the >> operator to read data from an input stream into a MappedTensor object.
     *
     * @param is An input stream (host-to-host) from which the data will be read.
     * @param tensor A reference (host-to-host) to the MappedTensor object where the data will be stored.
     *
     * @return A reference to the input stream `is` after the reading operation.
     *
     * This function provides a convenient way to populate a MappedTensor object with data from an input stream. It iterates through the elements of the tensor and reads values from the input stream one by one, until either all elements of the tensor have been filled or the input stream fails to provide more data.
     *
     * Memory management: The function does not allocate or deallocate any memory. It assumes that the `_data` array of the MappedTensor has already been allocated with the appropriate size (`_size`).
     * Exception handling: If the input stream fails to provide data (e.g., due to end-of-file or an invalid input format), the loop will terminate, and the function will return the input stream in its current state. No exceptions are thrown by this function itself, but the `>>` operator on the input stream may throw exceptions depending on its implementation.
     * Relationship with other components: This function is related to the data input component of the MappedTensor. It integrates with the standard input stream to allow easy data population.
     *
     * @note
     * - The time complexity of this function is O(n), where n is the size of the tensor (`_size`), as it iterates through each element of the tensor once.
     * - Ensure that the input stream contains valid data in the correct format to avoid unexpected behavior.
     *
     * @code
     * ```cpp
     * nz::data::MappedTensor::shape_type shape = {2, 3};
     * nz::data::MappedTensor tensor(shape, false);
     * std::istringstream iss("1 2 3 4 5 6");
     * iss >> tensor;
     * ```
     * @endcode
     */
    std::istream& operator>>(std::istream& is, MappedTensor& tensor) {
        MappedTensor::size_type i = 0;
        while (i < tensor._size && is >> tensor._data[i++]) {
        }
        return is;
    }

    MappedTensor::MappedTensor(const shape_type& shape, bool requires_grad) : _shape(shape),
                                                                              _size(shape[0] * shape[1]),
                                                                              _requires_grad(requires_grad) {
        CHECK(cudaMallocHost(&_data, _size * sizeof(value_type)));
        if (requires_grad) {
            CHECK(cudaMallocHost(&_grad, _size * sizeof(value_type)));
        }
        else {
            _grad = nullptr;
        }
    }

    MappedTensor::MappedTensor() : MappedTensor({0, 0}, false) {
        _data = nullptr;
        _grad = nullptr;
    }

    MappedTensor::MappedTensor(const MappedTensor& other) : MappedTensor(other._shape, other._requires_grad) {
        cuStrm::StreamManager<value_type>::Instance().memcpy(_data, other._data, _size * sizeof(value_type),
                                                             cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().memcpy(_grad, other._grad, _size * sizeof(value_type),
                                                                 cudaMemcpyDeviceToDevice);
        }
    }

    MappedTensor::MappedTensor(MappedTensor&& other) noexcept {
        _shape = std::move(other._shape);
        _size = other._size;
        _requires_grad = other._requires_grad;
        _data = other._data;
        _grad = other._grad;
        other._data = nullptr;
        other._grad = nullptr;
        other._size = 0;
        other._requires_grad = false;
        other._shape = {0, 0};
    }

    MappedTensor& MappedTensor::operator=(const MappedTensor& other) {
        if (this != &other) {
            if (_requires_grad && _grad) {
                cuStrm::StreamManager<value_type>::Instance().freeHost(_grad);
            }
            if (_data) {
                cuStrm::StreamManager<value_type>::Instance().freeHost(_data);
            }
            _shape = other._shape;
            _size = other._size;
            _requires_grad = other._requires_grad;
            CHECK(cudaMallocHost(&_data, _size * sizeof(value_type)));
            if (_requires_grad) {
                CHECK(cudaMallocHost(&_grad, _size * sizeof(value_type)));
            }
            else {
                _grad = nullptr;
            }
            cuStrm::StreamManager<value_type>::Instance().memcpy(_data, other._data, _size * sizeof(value_type),
                                                                 cudaMemcpyDeviceToDevice);
            if (_requires_grad) {
                cuStrm::StreamManager<value_type>::Instance().memcpy(_grad, other._grad, _size * sizeof(value_type),
                                                                     cudaMemcpyDeviceToDevice);
            }
            return *this;
        }
        return *this;
    }

    MappedTensor& MappedTensor::operator=(MappedTensor&& other) noexcept(false) {
        if (this != &other) {
            if (_data) {
                cuStrm::StreamManager<value_type>::Instance().freeHost(_data);
            }
            if (_requires_grad && _grad) {
                cuStrm::StreamManager<value_type>::Instance().freeHost(_grad);
            }
            _shape = std::move(other._shape);
            _size = other._size;
            _requires_grad = other._requires_grad;
            _data = other._data;
            _grad = other._grad;
            other._data = nullptr;
            other._grad = nullptr;
            other._size = 0;
            other._requires_grad = false;
            other._shape = {0, 0};
            return *this;
        }
        return *this;
    }

    MappedTensor::~MappedTensor() noexcept(false) {
        if (_requires_grad && _grad) {
            cuStrm::StreamManager<value_type>::Instance().freeHost(_grad);
        }
        if (_data) {
            cuStrm::StreamManager<value_type>::Instance().freeHost(_data);
        }
    }

    MappedTensor::iterator MappedTensor::begin() const {
        sync();
        return _data;
    }

    MappedTensor::iterator MappedTensor::end() const {
        sync();
        return _data + _size;
    }

    bool MappedTensor::requiresGrad() const noexcept {
        return _requires_grad;
    }

    MappedTensor::value_type* MappedTensor::data() const noexcept {
        return _data;
    }

    MappedTensor::size_type MappedTensor::size() const noexcept {
        return _size;
    }

    MappedTensor::shape_type MappedTensor::shape() const noexcept {
        return _shape;
    }

    void MappedTensor::setRequiresGrad(const bool requires_grad) {
        if (_requires_grad && !requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().freeHost(_grad);
            _grad = nullptr;
            _requires_grad = requires_grad;
        }
        if (!_requires_grad && requires_grad) {
            CHECK(cudaMallocHost(&_grad, _size * sizeof(value_type)));
            _requires_grad = requires_grad;
        }
    }

    void MappedTensor::setShape(const shape_type& shape) {
        value_type* temp;
        CHECK(cudaMallocHost(&temp, shape[0] * shape[1] * sizeof(value_type)));
        cuStrm::StreamManager<value_type>::Instance().memset(temp, 0, shape[0] * shape[1] * sizeof(value_type));
        cuStrm::StreamManager<value_type>::Instance().memcpy(temp, _data,
                                                             (_size < (shape[0] * shape[1])
                                                                  ? _size
                                                                  : shape[0] * shape[1]) * sizeof(value_type)
                                                             ,
                                                             cudaMemcpyDeviceToDevice);
        cuStrm::StreamManager<value_type>::Instance().freeHost(_data);
        _data = temp;
        if (_requires_grad) {
            value_type* tempGrad;
            CHECK(cudaMallocHost(&tempGrad, shape[0] * shape[1] * sizeof(value_type)));
            cuStrm::StreamManager<value_type>::Instance().memset(tempGrad, 0, shape[0] * shape[1] * sizeof(value_type));
            cuStrm::StreamManager<value_type>::Instance().memcpy(tempGrad, _grad,
                                                                 (_size < (shape[0] * shape[1])
                                                                      ? _size
                                                                      : shape[0] * shape[1]) * sizeof(
                                                                     value_type),
                                                                 cudaMemcpyDeviceToDevice);
            cuStrm::StreamManager<value_type>::Instance().freeHost(_grad);
            _grad = tempGrad;
        }
        _shape = shape;
        _size = shape[0] * shape[1];
    }

    void MappedTensor::dataInject(float* data, const size_type size, const bool isGrad) const {
        if (isGrad) {
            if (!_requires_grad) {
                throw std::invalid_argument(
                    "Gradient injection is not allowed for tensors that do not require gradients.");
            }
            cuStrm::StreamManager<value_type>::Instance().memcpy(_grad, data,
                                                                 (size < _size ? size : _size) * sizeof(value_type),
                                                                 cudaMemcpyHostToDevice);
        }
        else {
            cuStrm::StreamManager<value_type>::Instance().memcpy(_data, data,
                                                                 (size < _size ? size : _size) * sizeof(value_type),
                                                                 cudaMemcpyHostToDevice);
        }
    }

    void MappedTensor::dataInject(const std::initializer_list<value_type>& data, const bool isGrad) const {
        if (isGrad && !_requires_grad) {
            throw std::invalid_argument("Gradient injection is not allowed for tensors that do not require gradients.");
        }
        for (size_type i = 0; i < (data.size() < _size ? data.size() : _size); i++) {
            if (isGrad) {
                _grad[i] = *(data.begin() + i);
            }
            else {
                _data[i] = *(data.begin() + i);
            }
        }
    }

    std::ostream& MappedTensor::print(std::ostream& os) const {
        const std::ostream_iterator<value_type> oit(os, " ");
        syncData();
        for (auto i = 0; i < _shape[0]; i++) {
            const auto begin = _data + i * _shape[1];
            const auto end = begin + _shape[1];
            os << "[" << std::flush;
            std::copy(begin, end, oit);
            os << "]" << std::endl;
        }
        return os;
    }

    std::ostream& MappedTensor::printGrad(std::ostream& os) const {
        if (!_requires_grad) {
            throw std::invalid_argument("Gradient printing is not allowed for tensors that do not require gradients.");
        }
        const std::ostream_iterator<value_type> oit(os, " ");
        syncGrad();
        for (auto i = 0; i < _shape[0]; i++) {
            const auto begin = _grad + i * _shape[1];
            const auto end = begin + _shape[1];
            os << "[" << std::flush;
            std::copy(begin, end, oit);
            os << "]" << std::endl;
        }
        return os;
    }

    auto MappedTensor::operator[](const size_type index) const -> value_type& {
        syncData();
        if (index >= _size) {
            throw std::out_of_range("Index out of range.");
        }
        return _data[index];
    }

    void MappedTensor::clear() const {
        cuStrm::StreamManager<value_type>::Instance().memset(_data, 0, _size * sizeof(value_type));
    }

    void MappedTensor::clearGrad() const {
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().memset(_grad, 0, _size * sizeof(value_type));
        }
        else {
            throw std::runtime_error("Gradient clearing is not allowed for tensors that do not require gradients.");
        }
    }

    void MappedTensor::reshape(const shape_type& shape) {
        const size_type newSize = shape[0] * shape[1];
        if (newSize != _size) {
            WARN("Reshaping to a different size will cause data loss");
        }
        value_type* temp;
        CHECK(cudaMallocHost(&temp, newSize * sizeof(value_type)));
        cuStrm::StreamManager<value_type>::Instance().memset(temp, 0, newSize * sizeof(value_type));
        cuStrm::StreamManager<value_type>::Instance().memcpy(temp, _data,
                                                             (_size < newSize ? _size : newSize) * sizeof(value_type),
                                                             cudaMemcpyDeviceToDevice);
        cuStrm::StreamManager<value_type>::Instance().freeHost(_data);
        _data = temp;
        if (_requires_grad) {
            value_type* tempGrad;
            CHECK(cudaMallocHost(&tempGrad, newSize * sizeof(value_type)));
            cuStrm::StreamManager<value_type>::Instance().memset(tempGrad, 0, newSize * sizeof(value_type));
            cuStrm::StreamManager<value_type>::Instance().memcpy(tempGrad, _grad,
                                                                 (_size < newSize ? _size : newSize) * sizeof(
                                                                     value_type), cudaMemcpyDeviceToDevice);
            cuStrm::StreamManager<value_type>::Instance().freeHost(_grad);
            _grad = tempGrad;
        }
        _shape = shape;
        _size = newSize;
    }

    void MappedTensor::randomize(size_type seed, const bool isGrad) const {
        if (isGrad && !_requires_grad) {
            throw std::invalid_argument(
                "Gradient randomization is not allowed for tensors that do not require gradients.");
        }
        if (!seed) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }
        curandGenerator_t generator;
        curandStatus_t status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        if (status != CURAND_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create CURAND generator.");
        }
        status = curandSetPseudoRandomGeneratorSeed(generator, seed);
        if (status != CURAND_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set CURAND seed.");
        }
        status = curandGenerateUniform(generator, isGrad ? _grad : _data, _size);
        if (status != CURAND_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to generate random numbers.");
        }
    }

    void MappedTensor::fill(const value_type value, const bool isGrad) const {
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::Fill(grid, block, isGrad ? _grad : _data, value, _size);
    }

    void MappedTensor::transpose() {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((_shape[0] + block.x - 1) / block.x, (_shape[1] + block.y - 1) / block.y);
        value_type* temp;
        CHECK(cudaMallocHost(&temp, _size * sizeof(value_type)));
        krnl::Transpose(grid, block, _data, temp, _shape[0], _shape[1]);
        cuStrm::StreamManager<value_type>::Instance().freeHost(_data);
        _data = temp;
        if (_requires_grad) {
            value_type* tempGrad;
            CHECK(cudaMallocHost(&tempGrad, _size * sizeof(value_type)));
            krnl::Transpose(grid, block, _grad, tempGrad, _shape[0], _shape[1]);
            CHECK(cudaDeviceSynchronize());
            cuStrm::StreamManager<value_type>::Instance().freeHost(_grad);
            _grad = tempGrad;
        }
        std::swap(_shape[0], _shape[1]);
    }

    MappedTensor MappedTensor::operator+(const MappedTensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Shapes must be equal.");
        }
        const MappedTensor result(_shape, _requires_grad || other._requires_grad);
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::MatrixAdd(grid, block, _data, other._data, result._data, _size);
        return result;
    }

    MappedTensor MappedTensor::operator-(const MappedTensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Shapes must be equal.");
        }
        const MappedTensor result(_shape, _requires_grad || other._requires_grad);
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::MatrixSub(grid, block, _data, other._data, result._data, _size);
        return result;
    }

    MappedTensor MappedTensor::operator*(const MappedTensor& other) const {
        if (_shape[1] != other._shape[0]) {
            throw std::invalid_argument("Matrix shapes do not match");
        }
        MappedTensor result({_shape[0], other._shape[1]}, _requires_grad);
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((result._shape[1] + block.x - 1) / block.x, (result._shape[0] + block.y - 1) / block.y);
        krnl::GeneralMatrixMul(grid, block, _data, other._data, result._data, _shape[0], other._shape[1],
                               _shape[1]);
        return result;
    }

    MappedTensor MappedTensor::operator-() const {
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        MappedTensor result(_shape, _requires_grad);
        krnl::Negation(grid, block, result._data, _data, _size);
        return result;
    }

    MappedTensor MappedTensor::operator/(const MappedTensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Shapes must be equal.");
        }
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        MappedTensor result(_shape, _requires_grad);
        krnl::ElementwiseDivide(grid, block, result._data, _data, other._data, _size);
        return result;
    }

    void MappedTensor::recip() {
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        value_type* temp;
        CHECK(cudaMallocHost(&temp, _size * sizeof(value_type)));
        krnl::Recip(grid, block, temp, _data, _size);
        cuStrm::StreamManager<value_type>::Instance().freeHost(_data);
        _data = temp;
    }

    MappedTensor::value_type MappedTensor::sum() const {
        const dim3 block(256);
        const dim3 grid((_size + block.x - 1) / block.x);
        value_type* dData;
        CHECK(cudaMallocHost(&dData, grid.x * sizeof(value_type)));
        krnl::Summation(grid, block, block.x / WARP_SIZE * sizeof(float), dData, _data, _size);
        value_type result = 0;
        for (auto i = 0; i < grid.x; ++i) {
            result += dData[i];
        }
        cuStrm::StreamManager<value_type>::Instance().freeHost(dData);
        return result;
    }

    MappedTensor::value_type MappedTensor::expSum() const {
        const dim3 block(256);
        const dim3 grid((_size + block.x - 1) / block.x);
        value_type* dData;
        CHECK(cudaMallocHost(&dData, grid.x * sizeof(value_type)));
        krnl::SummationExp(grid, block, block.x / WARP_SIZE * sizeof(float), dData, _data, _size);
        value_type result = 0;
        for (auto i = 0; i < grid.x; ++i) {
            result += dData[i];
        }
        cuStrm::StreamManager<value_type>::Instance().freeHost(dData);
        return result;
    }

    void MappedTensor::syncGrad() const {
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().syncData(_grad);
        }
    }

    void MappedTensor::syncData() const {
        cuStrm::StreamManager<value_type>::Instance().syncData(_data);
    }

    void MappedTensor::sync() const {
        syncData();
        syncGrad();
    }
}

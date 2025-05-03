#include "NeuZephyr/MappedTensor.cuh"

#include <chrono>
#include <iterator>
#include <iostream>
#include <curand.h>

#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/utils.cuh"
#include "NeuZephyr/StreamManager.cuh"
#include "NeuZephyr/TensorOperations.cuh"

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
                                                                              _size(shape.size()),
                                                                              _requires_grad(requires_grad) {
        CHECK(cudaMallocHost(&_data, _size * sizeof(value_type)));
        if (requires_grad) {
            CHECK(cudaMallocHost(&_grad, _size * sizeof(value_type)));
        }
        else {
            _grad = nullptr;
        }
    }

    MappedTensor::MappedTensor() : MappedTensor({0, 0, 0, 0}, false) {
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
        _shape = other._shape;
        _size = other._size;
        _requires_grad = other._requires_grad;
        _data = other._data;
        _grad = other._grad;
        other._data = nullptr;
        other._grad = nullptr;
        other._size = 0;
        other._requires_grad = false;
        other._shape = {0, 0, 0, 0};
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
            _shape = other._shape;
            _size = other._size;
            _requires_grad = other._requires_grad;
            _data = other._data;
            _grad = other._grad;
            other._data = nullptr;
            other._grad = nullptr;
            other._size = 0;
            other._requires_grad = false;
            other._shape = {0, 0, 0, 0};
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

    MappedTensor::value_type* MappedTensor::grad() const {
        if (!_requires_grad) {
            throw std::invalid_argument("Gradient access is not allowed for tensors that do not require gradients.");
        }
        return _grad;
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
        _shape.reshape(shape);
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
        const auto& n = _shape.N();
        const auto& c = _shape.C();
        const auto& h = _shape.H();
        const auto& w = _shape.W();

        for (auto ni = 0; ni < n; ++ni) {
            os << "n=" << ni << " [\n";
            for (auto ci = 0; ci < c; ++ci) {
                os << "  c=" << ci << " [\n";
                for (auto hi = 0; hi < h; ++hi) {
                    const auto offset = ni * _shape.getStride(0) + ci * _shape.getStride(1) + hi * _shape.getStride(2);
                    const auto* begin = _data + offset;
                    const auto* end = begin + w;
                    os << "    [";
                    std::copy(begin, end, oit);
                    os << "]\n";
                }
                os << "  ]\n";
            }
            os << "]\n\n";
        }
        return os;
    }


    std::ostream& MappedTensor::printGrad(std::ostream& os) const {
        if (!_requires_grad) {
            throw std::invalid_argument("Gradient printing is not allowed for tensors that do not require gradients.");
        }
        const std::ostream_iterator<value_type> oit(os, " ");
        syncGrad();
        const auto& n = _shape.N();
        const auto& c = _shape.C();
        const auto& h = _shape.H();
        const auto& w = _shape.W();

        for (auto ni = 0; ni < n; ++ni) {
            os << "n=" << ni << " [\n";
            for (auto ci = 0; ci < c; ++ci) {
                os << "  c=" << ci << " [\n";
                for (auto hi = 0; hi < h; ++hi) {
                    const auto offset = ni * _shape.getStride(0) + ci * _shape.getStride(1) + hi * _shape.getStride(2);
                    const auto* begin = _grad + offset;
                    const auto* end = begin + w;
                    os << "    [";
                    std::copy(begin, end, oit);
                    os << "]\n";
                }
                os << "  ]\n";
            }
            os << "]\n\n";
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
        const size_type newSize = shape.size();
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
        if (isGrad && !_requires_grad) {
            throw std::invalid_argument(
                "Gradient filling is not allowed for tensors that do not require gradients.");
        }
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::Fill(grid, block, isGrad ? _grad : _data, value, _size);
    }

    void MappedTensor::fillMatrix(value_type value, size_type batch, size_type channels, bool isGrad) {
        if (batch >= _shape[0] || channels >= _shape[1]) {
            throw std::invalid_argument("Invalid batch or channels");
        }
        if (isGrad && !_requires_grad) {
            throw std::invalid_argument(
                "Gradient filling is not allowed for tensors that do not require gradients.");
        }
        const dim3 block(512);
        const dim3 grid((_shape[2] * _shape[3] + block.x - 1) / block.x);
        const auto offset = batch * _shape.getStride(0) + channels * _shape.getStride(1);
        krnl::Fill(grid, block, (isGrad ? _grad : _data), value, _shape[2] * _shape[3], offset);
    }

    void MappedTensor::transpose() {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((_shape[2] + block.x - 1) / block.x, (_shape[3] + block.y - 1) / block.y);
        value_type* temp;
        std::vector<size_t> offset;
        CHECK(cudaMallocHost(&temp, _size * sizeof(value_type)));
        for (auto i = 0; i < _shape[0]; i += 1) {
            for (auto j = 0; j < _shape[1]; j += 1) {
                offset.push_back(i * _shape.getStride(0) + j * _shape.getStride(1));
            }
        }
        krnl::Transpose(grid, block, _data, temp, _shape[2], _shape[3], offset);
        cuStrm::StreamManager<value_type>::Instance().freeHost(_data);
        _data = temp;
        if (_requires_grad) {
            value_type* tempGrad;
            CHECK(cudaMallocHost(&tempGrad, _size * sizeof(value_type)));
            krnl::Transpose(grid, block, _grad, tempGrad, _shape[2], _shape[3], offset);
            cuStrm::StreamManager<value_type>::Instance().freeHost(_grad);
            _grad = tempGrad;
        }
        std::swap(_shape[2], _shape[3]);
        _shape.updateStride();
    }

    MappedTensor MappedTensor::operator+(const MappedTensor& other) const {
        MappedTensor result(_shape.Broadcast(other._shape), _requires_grad || other._requires_grad);
        tensorMatrixAdd(result, *this, other);
        return result;
    }

    MappedTensor MappedTensor::operator-(const MappedTensor& other) const {
        MappedTensor result(_shape.Broadcast(other._shape), _requires_grad || other._requires_grad);
        tensorMatrixSub(result, *this, other);
        return result;
    }

    MappedTensor MappedTensor::operator*(const MappedTensor& other) const {
        MappedTensor result({
                                std::max(_shape.N(), other._shape.N()), std::max(_shape.C(), other._shape.C()),
                                _shape.H(), other._shape.W()
                            }, _requires_grad || other._requires_grad);
        tensorGeneralMatrixMul(result, *this, other);
        return result;
    }

    MappedTensor MappedTensor::operator-() const {
        MappedTensor result(_shape, _requires_grad);
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::Negation(grid, block, result._data, _data, _size);
        return result;
    }

    bool MappedTensor::operator==(const MappedTensor& other) const {
        if (_requires_grad != other._requires_grad) {
            return false;
        }
        if (_shape != other._shape) {
            return false;
        }
        this->sync();
        other.sync();
        for (auto i = 0; i < _size; i++) {
            if (_data[i] != other._data[i]) {
                return false;
            }
        }
        if (_requires_grad) {
            for (auto i = 0; i < _size; i++) {
                if (_grad[i] != other._grad[i]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool MappedTensor::operator!=(const MappedTensor& other) const {
        return !(*this == other);
    }

    MappedTensor MappedTensor::operator/(const MappedTensor& other) const {
        MappedTensor result(_shape.Broadcast(other._shape), _requires_grad || other._requires_grad);
        tensorElementwiseDivide(result, *this, other);
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
        cuStrm::StreamManager<value_type>::Instance().syncData(dData);
        for (auto i = 0; i < grid.x; ++i) {
            result += dData[i];
        }
        cuStrm::StreamManager<value_type>::Instance().freeHost(dData);
        return result;
    }

    MappedTensor::value_type MappedTensor::sum(const size_t batch, const size_t channel) const {
        if (batch >= _shape[0] || channel >= _shape[1]) {
            throw std::invalid_argument("Invalid position");
        }
        const auto size = _shape[2] * _shape[3];
        const dim3 block(256);
        const dim3 grid((size + block.x - 1) / block.x);
        value_type* dData;
        cudaMallocHost(&dData, grid.x * sizeof(value_type));
        const auto offset = batch * _shape.getStride(0) + channel * _shape.getStride(1);
        krnl::Summation(grid, block, block.x / WARP_SIZE * sizeof(float), dData, _data, size, offset);
        cuStrm::StreamManager<value_type>::Instance().syncData(dData);
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
        cuStrm::StreamManager<value_type>::Instance().syncData(dData);
        value_type result = 0;
        for (auto i = 0; i < grid.x; ++i) {
            result += dData[i];
        }
        cuStrm::StreamManager<value_type>::Instance().freeHost(dData);
        return result;
    }

    MappedTensor::value_type MappedTensor::expSum(const size_t batch, const size_t channel) const {
        if (batch >= _shape[0] || channel >= _shape[1]) {
            throw std::invalid_argument("Invalid position");
        }
        const auto size = _shape[2] * _shape[3];
        const dim3 block(256);
        const dim3 grid((size + block.x - 1) / block.x);
        value_type* dData;
        cudaMallocHost(&dData, grid.x * sizeof(value_type));
        const auto offset = batch * _shape.getStride(0) + channel * _shape.getStride(1);
        krnl::SummationExp(grid, block, block.x / WARP_SIZE * sizeof(float), dData, _data, size, offset);
        cuStrm::StreamManager<value_type>::Instance().syncData(dData);
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

#include "NeuZephyr/Tensor.cuh"
#include "NeuZephyr/utils.cuh"
#include "NeuZephyr/OperationKernels.cuh"
#include "NeuZephyr/NeuZephyrCudaErrorHandling.cuh"
#include "NeuZephyr/StreamManager.cuh"
#include "NeuZephyr/TensorOperations.cuh"
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
        tensor.print(os);
        if (tensor._requires_grad) {
            os << "Gradient: " << std::endl;
            tensor.printGrad(os);
        }
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
        cuStrm::StreamManager<float>::Instance().memcpy(tensor._data, data, tensor._size * sizeof(Tensor::value_type),
                                                        cudaMemcpyHostToDevice);
        free(data);
        return is;
    }

    // Constructors
    Tensor::Tensor() :
        _size(0), _shape({0, 0, 0, 0}), _data(nullptr), _grad(nullptr), _requires_grad(false) {
    }

    Tensor::Tensor(const shape_type& shape, const bool requires_grad) // NOLINT(*-pro-type-member-init)
        :
        _size(shape.size()), _shape(shape), _requires_grad(requires_grad) {
        cuStrm::StreamManager<value_type>::Instance().malloc(&_data, _size * sizeof(value_type));
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().malloc(&_grad, _size * sizeof(value_type));
        }
        else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const shape_type& shape, value_type* data, const bool requires_grad, const bool host) :
        _size(shape.size()), _shape(shape), _requires_grad(requires_grad) {
        cuStrm::StreamManager<value_type>::Instance().malloc(&_data, _size * sizeof(value_type));
        if (host) {
            cuStrm::StreamManager<value_type>::Instance().memcpy(_data, data, _size * sizeof(value_type),
                                                                 cudaMemcpyHostToDevice);
        }
        else {
            cuStrm::StreamManager<value_type>::Instance().memcpy(_data, data, _size * sizeof(value_type),
                                                                 cudaMemcpyDeviceToDevice);
        }
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().malloc(&_grad, _size * sizeof(value_type));
        }
        else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const shape_type& shape, const std::initializer_list<value_type>& data, const bool requires_grad) :
        _size(shape.size()), _shape(shape), _requires_grad(requires_grad) {
        if (std::distance(data.begin(), data.end()) < _size) {
            throw std::invalid_argument("Initializer list size is less than the tensor size.");
        }
        cuStrm::StreamManager<value_type>::Instance().malloc(&_data, _size * sizeof(value_type));
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().malloc(&_grad, _size * sizeof(value_type));
        }
        else {
            _grad = nullptr;
        }
        auto host_buf = new value_type[_size];
        auto it = data.begin();
        for (auto i = 0; i < _size; ++i, ++it) {
            host_buf[i] = *it;
        }
        cuStrm::StreamManager<value_type>::Instance().memcpy(_data, host_buf, _size * sizeof(value_type),
                                                             cudaMemcpyHostToDevice);
        delete[] host_buf;
    }

    // Copy and Move constructors
    Tensor::Tensor(const Tensor& other) :
        _size(other._size), _shape(other._shape), _requires_grad(other._requires_grad) {
        cuStrm::StreamManager<value_type>::Instance().malloc(&_data, _size * sizeof(value_type));
        cuStrm::StreamManager<value_type>::Instance().memcpy(_data, other._data, _size * sizeof(value_type),
                                                             cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().malloc(&_grad, _size * sizeof(value_type));
            cuStrm::StreamManager<value_type>::Instance().memcpy(_grad, other._grad, _size * sizeof(value_type),
                                                                 cudaMemcpyDeviceToDevice);
        }
        else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(Tensor&& other) noexcept(false):
        _size(other._size), _shape(other._shape), _requires_grad(other._requires_grad) {
        _data = other._data;
        if (_requires_grad) {
            _grad = other._grad;
        }
        other._data = nullptr;
        other._grad = nullptr;
        other._size = 0;
        other._shape = {0, 0, 0, 0};
    }

    Tensor& Tensor::operator=(const Tensor& other) {
        if (this != &other) {
            _size = other._size;
            _shape = other._shape;
            _requires_grad = other._requires_grad;
            cuStrm::StreamManager<value_type>::Instance().free(_data);
            cuStrm::StreamManager<value_type>::Instance().malloc(&_data, _size * sizeof(value_type));
            cuStrm::StreamManager<value_type>::Instance().memcpy(_data, other._data, _size * sizeof(value_type),
                                                                 cudaMemcpyDeviceToDevice);
            if (_requires_grad) {
                cuStrm::StreamManager<value_type>::Instance().free(_grad);
                cuStrm::StreamManager<value_type>::Instance().malloc(&_grad, _size * sizeof(value_type));
                cuStrm::StreamManager<value_type>::Instance().memcpy(_grad, other._grad, _size * sizeof(value_type),
                                                                     cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept(false) {
        if (this != &other) {
            _size = other._size;
            _shape = other._shape;
            cuStrm::StreamManager<value_type>::Instance().free(_data);
            _data = other._data;
            other._data = nullptr;
            if (_requires_grad) {
                cuStrm::StreamManager<value_type>::Instance().free(_grad);
                _grad = other._grad;
                other._grad = nullptr;
            }
            other._size = 0;
            other._shape = {0, 0, 0, 0};
        }
        return *this;
    }

    Tensor::~Tensor() noexcept(false) {
        if (_data != nullptr) {
            cuStrm::StreamManager<value_type>::Instance().freeAsync(_data);
            _data = nullptr;
        }
        if (_requires_grad) {
            if (_grad != nullptr) {
                cuStrm::StreamManager<value_type>::Instance().freeAsync(_grad);
                _grad = nullptr;
            }
        }
    }

    // Getter methods
    bool Tensor::requiresGrad() const noexcept { return _requires_grad; }
    Tensor::shape_type Tensor::shape() const noexcept { return _shape; }
    Tensor::size_type Tensor::size() const noexcept { return _size; }

    // Setter methods
    void Tensor::setRequiresGrad(const bool requires_grad) {
        if (requires_grad && _grad == nullptr) {
            cuStrm::StreamManager<value_type>::Instance().malloc(reinterpret_cast<value_type**>(_grad),
                                                                 _size * sizeof(value_type));
        }
        if (!requires_grad && _grad != nullptr) {
            cuStrm::StreamManager<value_type>::Instance().free(_grad);
            _grad = nullptr;
        }
        _requires_grad = requires_grad;
    }

    void Tensor::dataInject(const std::initializer_list<value_type>& data, const bool grad) const {
        dataInject(data.begin(), data.end(), grad);
    }

    // Operations
    void Tensor::zeroGrad() const {
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().memset(_grad, 0, _size * sizeof(value_type));
        }
    }

    std::ostream& Tensor::print(std::ostream& os) const {
        const std::ostream_iterator<value_type> output_iterator(os, " ");
        auto* data = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        cuStrm::StreamManager<value_type>::Instance().memcpy(data, _data, _size * sizeof(value_type),
                                                             cudaMemcpyDeviceToHost);
        cuStrm::StreamManager<value_type>::Instance().syncData(data);
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
                    const auto* begin = data + offset;
                    const auto* end = begin + w;
                    os << "    [";
                    std::copy(begin, end, output_iterator);
                    os << "]\n";
                }
                os << "  ]\n";
            }
            os << "]\n\n";
        }
        free(data);
        return os;
    }

    void Tensor::dataInject(value_type* data, const bool grad) const {
        if (grad) {
            if (_requires_grad) {
                cuStrm::StreamManager<value_type>::Instance().memcpy(_grad, data, _size * sizeof(value_type),
                                                                     cudaMemcpyHostToDevice);
            }
            else {
                throw std::runtime_error("Tensor does not require gradients");
            }
        }
        else {
            cuStrm::StreamManager<value_type>::Instance().memcpy(_data, data, _size * sizeof(value_type),
                                                                 cudaMemcpyHostToDevice);
        }
    }

    void Tensor::randomize(const unsigned long long seed) const {
        cuStrm::StreamManager<value_type>::Instance().randomize(_data, _size, seed, CURAND_RNG_PSEUDO_DEFAULT);
    }

    void Tensor::clear() const {
        cuStrm::StreamManager<value_type>::Instance().memset(_data, 0, _size * sizeof(value_type));
    }

    void Tensor::fill(const value_type value, const bool isGrad) const {
        if (isGrad && !_requires_grad) {
            throw std::invalid_argument(
                "Gradient filling is not allowed for tensors that do not require gradients.");
        }
        const dim3 block(512);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::Fill(grid, block, isGrad ? _grad : _data, value, _size);
    }

    void Tensor::fillMatrix(const value_type value, const size_type batch, const size_type channels,
                            const bool isGrad) {
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

    Tensor Tensor::operator+(const Tensor& other) const {
        Tensor result(_shape.Broadcast(other._shape), _requires_grad || other._requires_grad);
        tensorMatrixAdd(result, *this, other);
        return result;
    }

    Tensor Tensor::operator-(const Tensor& other) const {
        Tensor result(_shape.Broadcast(other._shape), _requires_grad || other._requires_grad);
        tensorMatrixSub(result, *this, other);
        return result;
    }

    Tensor Tensor::operator*(const Tensor& other) const {
        Tensor result({
                          std::max(_shape.N(), other._shape.N()), std::max(_shape.C(), other._shape.C()), _shape.H(),
                          other._shape.W()
                      }, _requires_grad || other._requires_grad);
        tensorGeneralMatrixMul(result, *this, other);
        return result;
    }

    Tensor Tensor::operator/(const Tensor& other) const {
        Tensor result(_shape.Broadcast(other._shape), _requires_grad || other._requires_grad);
        tensorElementwiseDivide(result, *this, other);
        return result;
    }

    void Tensor::reshape(const shape_type& shape) {
        const size_type size = shape.size();
        if (size != _size) {
            WARN("Reshaping to a different size will cause data loss");
        }
        value_type* temp;
        cuStrm::StreamManager<value_type>::Instance().malloc(&temp, size * sizeof(value_type));
        cuStrm::StreamManager<value_type>::Instance().memset(temp, 0, size * sizeof(value_type));
        cuStrm::StreamManager<value_type>::Instance().memcpy(temp, _data,
                                                             (size < _size ? size : _size) * sizeof(value_type),
                                                             cudaMemcpyDeviceToDevice);
        cuStrm::StreamManager<value_type>::Instance().free(_data);
        _data = temp;
        if (_requires_grad) {
            value_type* tempGrad;
            cuStrm::StreamManager<value_type>::Instance().malloc(&tempGrad, size * sizeof(value_type));
            cuStrm::StreamManager<value_type>::Instance().memset(tempGrad, 0, size * sizeof(value_type));
            cuStrm::StreamManager<value_type>::Instance().memcpy(tempGrad, _grad,
                                                                 (size < _size ? size : _size) * sizeof(value_type),
                                                                 cudaMemcpyDeviceToDevice);
            cuStrm::StreamManager<value_type>::Instance().free(_grad);
            _grad = tempGrad;
        }
        _shape = shape;
        _size = size;
    }

    void Tensor::transpose() {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((_shape[2] + block.x - 1) / block.x, (_shape[3] + block.y - 1) / block.y);
        value_type* temp;
        std::vector<size_t> offset;
        cuStrm::StreamManager<float>::Instance().malloc(&temp, _size * sizeof(value_type));
        for (auto i = 0; i < _shape[0]; i += 1) {
            for (auto j = 0; j < _shape[1]; j += 1) {
                offset.push_back(i * _shape.getStride(0) + j * _shape.getStride(1));
            }
        }
        krnl::Transpose(grid, block, _data, temp, _shape[2], _shape[3], offset);
        cuStrm::StreamManager<value_type>::Instance().freeAsync(_data);
        _data = temp;
        if (_requires_grad) {
            value_type* tempGrad;
            cuStrm::StreamManager<float>::Instance().malloc(&tempGrad, _size * sizeof(value_type));
            krnl::Transpose(grid, block, _grad, tempGrad, _shape[2], _shape[3], offset);
            cuStrm::StreamManager<value_type>::Instance().freeAsync(_grad);
            _grad = tempGrad;
        }
        std::swap(_shape[2], _shape[3]);
        _shape.updateStride();
    }

    void Tensor::setData(const shape_type& position, const value_type value, const bool isGrad) const {
        if (position[0] >= _shape[0] || position[1] >= _shape[1] || position[2] >= _shape[2] || position[3] >= _shape[
            3]) {
            throw std::invalid_argument("Invalid position");
        }
        if (isGrad && !_requires_grad) {
            throw std::invalid_argument(
                "Gradient setting is not allowed for tensors that do not require gradients.");
        }
        auto* data = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        cuStrm::StreamManager<value_type>::Instance().memcpy(data, isGrad ? _grad : _data, _size * sizeof(value_type),
                                                             cudaMemcpyDeviceToHost);
        cuStrm::StreamManager<value_type>::Instance().syncData(data);
        data[position[0] * _shape.getStride(0) + position[1] * _shape.getStride(1) + position[2] * _shape.getStride(2) +
            position[3] * _shape.getStride(3)] = value;
        cuStrm::StreamManager<value_type>::Instance().memcpy(isGrad ? _grad : _data, data, _size * sizeof(value_type),
                                                             cudaMemcpyHostToDevice);
        cuStrm::StreamManager<value_type>::Instance().syncData(isGrad ? _grad : _data);
        free(data);
    }


    Tensor::value_type* Tensor::data() const noexcept {
        return _data;
    }

    std::vector<Tensor::value_type> Tensor::hostData() const noexcept {
        auto temp = new value_type[_size];
        syncData();
        cudaMemcpy(temp, _data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        std::vector result(temp, temp + _size);
        delete[] temp;
        return result;
    }

    Tensor::value_type* Tensor::grad() const {
        if (!_requires_grad) {
            throw std::runtime_error("Tensor does not require gradients");
        }
        return _grad;
    }

    std::vector<Tensor::value_type> Tensor::hostGrad() const {
        if (!_requires_grad) {
            throw std::runtime_error("Tensor does not require gradients");
        }
        auto temp = new value_type[_size];
        syncGrad();
        cudaMemcpy(temp, _grad, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        std::vector result(temp, temp + _size);
        delete[] temp;
        return result;
    }

    std::ostream& Tensor::printGrad(std::ostream& os) const {
        if (!_requires_grad) {
            throw std::runtime_error("Tensor does not require gradients");
        }
        auto* data = static_cast<value_type*>(malloc(_size * sizeof(value_type)));
        cuStrm::StreamManager<value_type>::Instance().memcpy(data, _grad, _size * sizeof(value_type),
                                                             cudaMemcpyDeviceToHost);
        const std::ostream_iterator<value_type> output_iterator(os, " ");
        cuStrm::StreamManager<value_type>::Instance().syncData(data);
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
                    const auto* begin = data + offset;
                    const auto* end = begin + w;
                    os << "    [";
                    std::copy(begin, end, output_iterator);
                    os << "]\n";
                }
                os << "  ]\n";
            }
            os << "]\n\n";
        }
        free(data);
        return os;
    }

    Tensor Tensor::operator-() const {
        Tensor result(_shape, _requires_grad);
        const dim3 block(256);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::Negation(grid, block, result._data, _data, _size);
        return result;
    }

    bool Tensor::operator==(const Tensor& other) const {
        if (_requires_grad != other._requires_grad) {
            return false;
        }
        if (_shape != other._shape) {
            return false;
        }
        constexpr value_type abs_epsilon = 1e-6f;
        constexpr value_type rel_epsilon = 1e-5f;
        this->sync();
        other.sync();
        auto* temp = new value_type[_size];
        auto* temp_other = new value_type[_size];
        cudaMemcpy(temp, _data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_other, other._data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        for (auto i = 0; i < _size; i++) {
            const auto a = temp[i];
            const auto b = temp_other[i];
            const auto diff = std::abs(a - b);
            if (diff > std::max(rel_epsilon * std::max(std::abs(a), std::abs(b)), abs_epsilon)) {
                delete[] temp;
                delete[] temp_other;
                return false;
            }
        }
        if (_requires_grad) {
            cudaMemcpy(temp, _grad, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
            cudaMemcpy(temp_other, other._grad, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
            for (auto i = 0; i < _size; i++) {
                const auto a = temp[i];
                const auto b = temp_other[i];
                const auto diff = std::abs(a - b);
                if (diff > std::max(rel_epsilon * std::max(std::abs(a), std::abs(b)), abs_epsilon)) {
                    delete[] temp;
                    delete[] temp_other;
                    return false;
                }
            }
        }
        delete[] temp;
        delete[] temp_other;
        return true;
    }

    bool Tensor::operator!=(const Tensor& other) const {
        return !(*this == other);
    }

    void Tensor::recip() const {
        value_type* data;
        cuStrm::StreamManager<value_type>::Instance().malloc(&data, _size * sizeof(value_type));
        const dim3 block(256);
        const dim3 grid((_size + block.x - 1) / block.x);
        krnl::Recip(grid, block, data, _data, _size);
        cuStrm::StreamManager<value_type>::Instance().memcpy(_data, data, _size * sizeof(value_type),
                                                             cudaMemcpyDeviceToDevice);
        cuStrm::StreamManager<value_type>::Instance().free(data);
    }

    Tensor::value_type Tensor::sum() const {
        const dim3 block(256);
        const dim3 grid((_size + block.x - 1) / block.x);
        value_type* dData;
        auto* hData = new value_type[grid.x];
        cuStrm::StreamManager<value_type>::Instance().malloc(&dData, grid.x * sizeof(value_type));
        krnl::Summation(grid, block, block.x / WARP_SIZE * sizeof(float), dData, _data, _size);
        cuStrm::StreamManager<value_type>::Instance().memcpy(hData, dData, grid.x * sizeof(value_type),
                                                             cudaMemcpyDeviceToHost);
        cuStrm::StreamManager<value_type>::Instance().syncData(hData);
        value_type result = 0;
        for (auto i = 0; i < grid.x; ++i) {
            result += hData[i];
        }
        delete[] hData;
        cuStrm::StreamManager<value_type>::Instance().free(dData);
        return result;
    }

    Tensor::value_type Tensor::sum(const size_type batch, const size_type channel) const {
        if (batch >= _shape[0] || channel >= _shape[1]) {
            throw std::invalid_argument("Invalid position");
        }
        const auto size = _shape[2] * _shape[3];
        const dim3 block(256);
        const dim3 grid((size + block.x - 1) / block.x);
        value_type* dData;
        auto* hData = new value_type[grid.x];
        cuStrm::StreamManager<value_type>::Instance().malloc(&dData, grid.x * sizeof(value_type));
        const auto offset = batch * _shape.getStride(0) + channel * _shape.getStride(1);
        krnl::Summation(grid, block, block.x / WARP_SIZE * sizeof(float), dData, _data, size, offset);
        cuStrm::StreamManager<value_type>::Instance().memcpy(hData, dData, grid.x * sizeof(value_type),
                                                             cudaMemcpyDeviceToHost);
        cuStrm::StreamManager<value_type>::Instance().syncData(hData);
        value_type result = 0;
        for (auto i = 0; i < grid.x; ++i) {
            result += hData[i];
        }
        delete[] hData;
        cuStrm::StreamManager<value_type>::Instance().free(dData);
        return result;
    }

    Tensor::value_type Tensor::max() const {
        auto hData = hostData();
        value_type result = std::numeric_limits<value_type>::min();
        for (auto i = 0; i < _size; ++i) {
            if (hData[i] > result) {
                result = hData[i];
            }
        }
        return result;
    }

    Tensor::value_type Tensor::max(const size_type batch, const size_type channel) const {
        if (batch >= _shape[0] || channel >= _shape[1]) {
            throw std::invalid_argument("Invalid position");
        }
        const auto offset = batch * _shape.getStride(0) + channel * _shape.getStride(1);
        auto hData = hostData();
        value_type result = std::numeric_limits<value_type>::min();
        for (auto i = 0; i < _shape[2] * _shape[3]; ++i) {
            if (hData[offset + i] > result) {
                result = hData[offset + i];
            }
        }
        return result;
    }

    Tensor::value_type Tensor::min() const {
        auto hData = hostData();
        value_type result = std::numeric_limits<value_type>::max();
        for (auto i = 0; i < _size; ++i) {
            if (hData[i] < result) {
                result = hData[i];
            }
        }
        return result;
    }

    Tensor::value_type Tensor::min(const size_type batch, const size_type channel) const {
        if (batch >= _shape[0] || channel >= _shape[1]) {
            throw std::invalid_argument("Invalid position");
        }
        const auto offset = batch * _shape.getStride(0) + channel * _shape.getStride(1);
        auto hData = hostData();
        value_type result = std::numeric_limits<value_type>::max();
        for (auto i = 0; i < _shape[2] * _shape[3]; ++i) {
            if (hData[offset + i] < result) {
                result = hData[offset + i];
            }
        }
        return result;
    }

    Tensor::shape_type Tensor::find(const value_type value) const {
        auto hData = hostData();
        auto index = 0;
        for (auto i = 0; i < _size; ++i) {
            if (hData[i] == value) {
                index = i;
                break;
            }
        }
        auto n = index / (_shape[1] * _shape[2] * _shape[3]);
        auto c = (index % (_shape[1] * _shape[2] * _shape[3])) / (_shape[2] * _shape[3]);
        auto h = (index % (_shape[2] * _shape[3])) / _shape[3];
        auto w = index % _shape[3];
        return {n, c, h, w};
    }

    Tensor::shape_type Tensor::find(value_type value, size_type batch, size_type channel) const {
        if (batch >= _shape[0] || channel >= _shape[1]) {
            throw std::invalid_argument("Invalid position");
        }
        auto hData = hostData();
        auto index = 0;
        auto offset = batch * _shape.getStride(0) + channel * _shape.getStride(1);
        for (auto i = 0; i < _shape[2] * _shape[3]; ++i) {
            if (hData[offset + i] == value) {
                index = i;
                break;
            }
        }
        auto h = index / _shape[3];
        auto w = index % _shape[3];
        return {batch, channel, h, w};
    }

    Tensor::value_type Tensor::expSum() const {
        const dim3 block(256);
        const dim3 grid((_size + block.x - 1) / block.x);
        value_type* dData;
        auto* hData = new value_type[grid.x];
        cuStrm::StreamManager<value_type>::Instance().malloc(&dData, grid.x * sizeof(value_type));
        krnl::SummationExp(grid, block, block.x / WARP_SIZE * sizeof(float), dData, _data, _size);
        cuStrm::StreamManager<value_type>::Instance().memcpy(hData, dData, grid.x * sizeof(value_type),
                                                             cudaMemcpyDeviceToHost);
        cuStrm::StreamManager<value_type>::Instance().syncData(hData);
        value_type result = 0;
        for (auto i = 0; i < grid.x; ++i) {
            result += hData[i];
        }
        delete[] hData;
        cuStrm::StreamManager<value_type>::Instance().free(dData);
        return result;
    }

    Tensor::value_type Tensor::expSum(const size_t batch, const size_t channel) const {
        if (batch >= _shape[0] || channel >= _shape[1]) {
            throw std::invalid_argument("Invalid position");
        }
        const auto size = _shape[2] * _shape[3];
        const dim3 block(256);
        const dim3 grid((size + block.x - 1) / block.x);
        value_type* dData;
        auto* hData = new value_type[grid.x];
        cuStrm::StreamManager<value_type>::Instance().malloc(&dData, grid.x * sizeof(value_type));
        const auto offset = batch * _shape.getStride(0) + channel * _shape.getStride(1);
        krnl::SummationExp(grid, block, block.x / WARP_SIZE * sizeof(float), dData, _data, size, offset);
        cuStrm::StreamManager<value_type>::Instance().memcpy(hData, dData, grid.x * sizeof(value_type),
                                                             cudaMemcpyDeviceToHost);
        cuStrm::StreamManager<value_type>::Instance().syncData(hData);
        value_type result = 0;
        for (auto i = 0; i < grid.x; ++i) {
            result += hData[i];
        }
        delete[] hData;
        cuStrm::StreamManager<value_type>::Instance().free(dData);
        return result;
    }

    void Tensor::syncData() const {
        cuStrm::StreamManager<value_type>::Instance().syncData(_data);
    }

    void Tensor::syncGrad() const {
        if (_requires_grad) {
            cuStrm::StreamManager<value_type>::Instance().syncData(_grad);
        }
    }

    void Tensor::sync() const {
        syncData();
        syncGrad();
    }
}

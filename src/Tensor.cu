#include "NeuZephyr/Tensor.cuh"

namespace NeuZephyr::Data {
    // Stream operators
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
        free(data);
        return os;
    }

    std::istream& operator>>(std::istream& is, const Tensor& tensor) {
        auto* data = static_cast<Tensor::value_type*>(malloc(tensor._size * sizeof(Tensor::value_type)));
        for (int i = 0; i < tensor._size; ++i) {
            is >> data[i];
        }
        cudaMemcpy(tensor._data, data, tensor._size * sizeof(Tensor::value_type), cudaMemcpyHostToDevice);
        free(data);
        return is;
    }

    Tensor operator*(const Tensor::value_type lhs, const Tensor& rhs) {
        Tensor result(rhs._shape, rhs._requires_grad);
        dim3 block(256);
        dim3 grid((rhs._size + block.x - 1) / block.x);
        Kernels::ScalarMul<<<grid, block>>>(result._data, rhs._data, lhs, rhs._size);
        return result;
    }

    Tensor operator*(const Tensor& lhs, const Tensor::value_type rhs) {
        Tensor result(lhs._shape, lhs._requires_grad);
        dim3 block(256);
        dim3 grid((lhs._size + block.x - 1) / block.x);
        Kernels::ScalarMul<<<grid, block>>>(result._data, lhs._data, rhs, lhs._size);
        return result;
    }

    Tensor operator/(const Tensor& lhs, const Tensor::value_type rhs) {
        Tensor result(lhs._shape, lhs._requires_grad);
        dim3 block(256);
        dim3 grid((lhs._size + block.x - 1) / block.x);
        Kernels::ScalarDiv<<<grid, block>>>(result._data, lhs._data, rhs, lhs._size);
        return result;
    };

    Tensor operator+(const Tensor& lhs, const Tensor::value_type rhs) {
        Tensor result(lhs._shape, lhs._requires_grad);
        dim3 block(256);
        dim3 grid((lhs._size + block.x - 1) / block.x);
        Kernels::ScalarAdd<<<grid, block>>>(result._data, lhs._data, rhs, lhs._size);
        return result;
    };

    Tensor operator+(const Tensor::value_type lhs, const Tensor& rhs) {
        Tensor result(rhs._shape, rhs._requires_grad);
        dim3 block(256);
        dim3 grid((rhs._size + block.x - 1) / block.x);
        Kernels::ScalarAdd<<<grid, block>>>(result._data, rhs._data, lhs, rhs._size);
        return result;
    };

    Tensor operator-(const Tensor& lhs, const Tensor::value_type rhs) {
        Tensor result(lhs._shape, lhs._requires_grad);
        dim3 block(256);
        dim3 grid((lhs._size + block.x - 1) / block.x);
        Kernels::ScalarAdd<<<grid, block>>>(result._data, lhs._data, -rhs, lhs._size);
        return result;
    };

    Tensor operator-(const Tensor::value_type lhs, const Tensor& rhs) {
        Tensor result(rhs._shape, rhs._requires_grad);
        dim3 block(256);
        dim3 grid((rhs._size + block.x - 1) / block.x);
        Kernels::ScalarAdd<<<grid, block>>>(result._data, rhs._data, -lhs, rhs._size);
        return result;
    };

    Tensor ReLU(const Tensor& tensor) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        Kernels::RectifiedLinearUnit<<<grid, block>>>(result._data, tensor._data, tensor._size);
        return result;
    }

    Tensor Sigmoid(const Tensor& tensor) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        Kernels::Sigmoid<<<grid, block>>>(result._data, tensor._data, tensor._size);
        return result;
    }

    Tensor Tanh(const Tensor& tensor) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        Kernels::Tanh<<<grid, block>>>(result._data, tensor._data, tensor._size);
        return result;
    }

    Tensor LeakyReLU(const Tensor& tensor, float alpha) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        Kernels::LeakyReLU<<<grid, block>>>(result._data, tensor._data, tensor._size, alpha);
        return result;
    }

    Tensor Swish(const Tensor& tensor) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        Kernels::Swish<<<grid, block>>>(result._data, tensor._data, tensor._size);
        return result;
    }

    Tensor ELU(const Tensor& tensor, float alpha) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        Kernels::ExponentialLinearUnit<<<grid, block>>>(result._data, tensor._data, tensor._size, alpha);
        return result;
    }

    Tensor HardSigmoid(const Tensor& tensor, float alpha, float beta) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        Kernels::HardSigmoid<<<grid, block>>>(result._data, tensor._data, tensor._size, alpha, beta);
        return result;
    }

    Tensor HardSwish(const Tensor& tensor, float alpha, float beta) {
        Tensor result(tensor._shape, tensor._requires_grad);
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        Kernels::HardSwish<<<grid, block>>>(result._data, tensor._data, tensor._size, alpha, beta);
        return result;
    }

    Tensor Softmax(const Tensor& tensor) {
        dim3 block(256);
        dim3 grid((tensor._size + block.x - 1) / block.x);
        float* result_d;
        float* result_h;
        float sum = 0;
        cudaMalloc(&result_d, grid.x * sizeof(Tensor::value_type));
        result_h = (float*)malloc(grid.x * sizeof(Tensor::value_type));
        Kernels::SummationExp<<<grid, block, block.x * sizeof(float)>>>(result_d, tensor._data, tensor._size);
        cudaMemcpy(result_h, result_d, grid.x * sizeof(Tensor::value_type), cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid.x; i++) {
            sum += result_h[i];
        }
        free(result_h);
        cudaFree(result_d);
        Kernels::Softmax<<<grid, block>>>(tensor._data, tensor._data, sum, tensor._size);
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

    Tensor::Tensor(const shape_type& shape, const value_type* data, const bool requires_grad) :
        _size(shape[0] * shape[1]), _shape(shape), _requires_grad(requires_grad) {
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        }
        else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const std::initializer_list<int>& shape, const bool requires_grad) :
        _shape(shape), _requires_grad(requires_grad) {
        _size = _shape[0] * _shape[1];
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        }
        else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const std::initializer_list<int>& shape, const value_type* data, const bool requires_grad) :
        _shape(shape), _requires_grad(requires_grad) {
        _size = _shape[0] * _shape[1];
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        }
        else {
            _grad = nullptr;
        }
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

    void Tensor::copyData(const value_type* data, const shape_type& shape) {
        cudaFree(_data);
        if (_requires_grad) {
            cudaFree(_grad);
        }
        _size = shape[0] * shape[1];
        _shape = shape;
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyHostToDevice);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
            cudaMemset(_grad, 0, _size * sizeof(value_type));
        }
    }

    void Tensor::copyGrad(const value_type* grad) const {
        if (!_requires_grad) {
            throw std::runtime_error("Tensor does not require gradients");
        }
        cudaMemcpy(_grad, grad, _size * sizeof(value_type), cudaMemcpyHostToDevice);
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


    // Arithmetic operators
    Tensor Tensor::operator+(const Tensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor result(_shape, _requires_grad);
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        Kernels::MatrixAdd<<<grid, block>>>(_data, other._data, result._data, _size);
        return result;
    }

    Tensor Tensor::operator-(const Tensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor result(_shape, _requires_grad);
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        Kernels::MatrixSub<<<grid, block>>>(_data, other._data, result._data, _size);
        return result;
    }

    Tensor Tensor::operator*(const Tensor& other) const {
        if (_shape[1] != other._shape[0]) {
            throw std::invalid_argument("Matrix shapes do not match");
        }
        Tensor result({_shape[0], other._shape[1]}, _requires_grad);
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((result._shape[1] + block.x - 1) / block.x, (result._shape[0] + block.y - 1) / block.y);
        Kernels::GeneralMatrixMul<<<grid, block>>>(_data, other._data, result._data, _shape[0], other._shape[1],
                                                   _shape[1]);
        return result;
    }

    void Tensor::reshape(const shape_type& shape) {
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
        Kernels::Transpose<<<grid, block>>>(temp, _data, _shape[0], _shape[1]);
        reshape({_shape[1], _shape[0]});
        cudaFree(temp);
    }

    void Tensor::setData(const shape_type& position, const value_type value) const {
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

    Tensor::value_type* Tensor::grad() const noexcept {
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
        Kernels::Negation<<<grid, block>>>(result._data, _data, _size);
        return result;
    }

    void Tensor::recip() const {
        value_type* data;
        cudaMalloc(reinterpret_cast<value_type**>(&data), _size * sizeof(value_type));
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        Kernels::Recip<<<grid, block>>>(data, _data, _size);
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        cudaFree(data);
    }


}

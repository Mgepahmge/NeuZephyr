#include "DL-Framework/Tensor.cuh"

namespace DL {
    // Stream operators
    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        Tensor::value_type* data = (Tensor::value_type*)malloc(tensor._size*sizeof(Tensor::value_type));
        cudaMemcpy(data, tensor._data, tensor._size*sizeof(Tensor::value_type), cudaMemcpyDeviceToHost);
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

    std::istream& operator>>(std::istream& is, Tensor& tensor) {
        Tensor::value_type* data = (Tensor::value_type*)malloc(tensor._size*sizeof(Tensor::value_type));
        for (int i = 0; i < tensor._size; ++i) {
            is >> data[i];
        }
        cudaMemcpy(tensor._data, data, tensor._size*sizeof(Tensor::value_type), cudaMemcpyHostToDevice);
        free(data);
        return is;
    }

    // Constructors
    Tensor::Tensor() : _size(0), _shape({0, 0}), _data(nullptr), _grad(nullptr), _requires_grad(false) {}

    Tensor::Tensor(const shape_type &shape, const bool requires_grad)
        : _size(shape[0] * shape[1]), _shape(shape), _requires_grad(requires_grad) {
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        } else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const shape_type &shape, const value_type* data, const bool requires_grad)
        : _size(shape[0] * shape[1]), _shape(shape), _requires_grad(requires_grad) {
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        } else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const std::initializer_list<int> &shape, const bool requires_grad)
        : _shape(shape), _requires_grad(requires_grad) {
        _size = _shape[0] * _shape[1];
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        } else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(const std::initializer_list<int> &shape, const value_type* data, const bool requires_grad)
        : _shape(shape), _requires_grad(requires_grad) {
        _size = _shape[0] * _shape[1];
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
        } else {
            _grad = nullptr;
        }
    }

    // Copy and Move constructors
    Tensor::Tensor(const Tensor& other)
        : _size(other._size), _shape(other._shape), _requires_grad(other._requires_grad) {
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemcpy(_data, other._data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
            cudaMemcpy(_grad, other._grad, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        } else {
            _grad = nullptr;
        }
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : _size(other._size), _shape(std::move(other._shape)), _requires_grad(other._requires_grad) {
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
    bool Tensor::requires_grad() const noexcept { return _requires_grad; }
    Tensor::shape_type Tensor::shape() const noexcept { return _shape; }
    Tensor::size_type Tensor::size() const noexcept { return _size; }

    // Setter methods
    void Tensor::set_requires_grad(const bool requires_grad) noexcept {
        if (requires_grad && _grad == nullptr) {
            cudaMalloc((value_type**)_grad, _size * sizeof(value_type));
        }
        if (!requires_grad && _grad != nullptr) {
            cudaFree(_grad);
            _grad = nullptr;
        }
        _requires_grad = requires_grad;
    }

    // Operations
    void Tensor::zero_grad() const noexcept {
        if (_requires_grad) {
            cudaMemset(_grad, 0, _size * sizeof(value_type));
        }
    }

    void Tensor::print() const noexcept {
        std::ostream_iterator<value_type> output_iterator(std::cout, " ");
        value_type* data = (value_type*)malloc(_size * sizeof(value_type));
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

    void Tensor::copy_data(const value_type* data, const shape_type &shape) {
        cudaFree(_data);
        if (_requires_grad) {
            cudaFree(_grad);
        }
        _size = shape[0] * shape[1];
        _shape = shape;
        cudaMalloc((value_type**)&_data, _size * sizeof(value_type));
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((value_type**)&_grad, _size * sizeof(value_type));
            cudaMemset(_grad, 0, _size * sizeof(value_type));
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
        cudaMemset(_data, value, _size * sizeof(value_type));
    }

    // Arithmetic operators
    Tensor Tensor::operator+(const Tensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor result(_shape, _requires_grad);
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        add_kernel<<<grid, block>>>(_data, other._data, result._data, _size);
        return result;
    }

    Tensor Tensor::operator-(const Tensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor result(_shape, _requires_grad);
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        sub_kernel<<<grid, block>>>(_data, other._data, result._data, _size);
        return result;
    }

    Tensor Tensor::operator*(const Tensor& other) const {
        if (_shape[1] != other._shape[0]) {
            throw std::invalid_argument("Matrix shapes do not match");
        }
        Tensor result({_shape[0], other._shape[1]}, _requires_grad);
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((_shape[0] + block.x - 1) / block.x, (_shape[1] + block.y - 1) / block.y);
        GEMM_kernel<<<grid, block>>>(_data, other._data, result._data, _shape[0], other._shape[1], _shape[1]);
        return result;
    }

    void Tensor::reshape(const shape_type &shape) {
        value_type* temp = (value_type*)malloc(_size * sizeof(value_type));
        cudaMemcpy(temp, _data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        cudaFree(_data);
        value_type* temp_grad = nullptr;
        if (_requires_grad) {
            temp_grad = (value_type*)malloc(_size * sizeof(value_type));
            cudaMemcpy(temp_grad, _grad, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
            cudaFree(_grad);
        }
        size_type size = _size;
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

    void Tensor::reshape(const std::initializer_list<int> &shape) {
        reshape(shape_type(shape));
    }

    void Tensor::transpose() {
        value_type* temp;
        cudaMalloc((value_type**)&temp, _size * sizeof(value_type));
        cudaMemcpy(temp, _data, _size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((_shape[0] + block.x - 1) / block.x, (_shape[1] + block.y - 1) / block.y);
        Transpose_kernel<<<grid, block>>>(temp, _data, _shape[0], _shape[1]);
        reshape({_shape[1], _shape[0]});
        cudaFree(temp);
    }

    void Tensor::set_data(const shape_type &position, const value_type value) const {
        value_type* data = (value_type*)malloc(_size * sizeof(value_type));
        cudaMemcpy(data, _data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        data[position[0] * _shape[1] + position[1]] = value;
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyHostToDevice);
        free(data);
    }

    void Tensor::set_data(const std::initializer_list<int>& position, const value_type value) const {
        set_data(shape_type(position), value);
    }
} //DL
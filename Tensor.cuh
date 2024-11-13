//
// Created by Administrator on 24-11-11.
//

#ifndef TENSOR_CUH
#define TENSOR_CUH
#include "OperationKernels.cuh"
#include <iterator>
#include <stdexcept>
#include <curand.h>
#include <vector>
#include <iostream>

namespace DL {

class Tensor {
public:
    using size_type = unsigned long long;
    using value_type = float;
    using shape_type = std::vector<int>;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    friend std::istream& operator>>(std::istream& is, Tensor& tensor);

    Tensor() : _size(0), _shape({0, 0}), _data(nullptr), _grad(nullptr), _requires_grad(false) {}

    explicit Tensor(const shape_type &shape, const bool requires_grad = false) : _size(shape[0] * shape[1]), _shape(shape), _requires_grad(requires_grad) {
        cudaMalloc((float**)&_data, _size * sizeof(float));
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
        } else {
            _grad = nullptr;
        }
    }

    explicit Tensor(const shape_type &shape, const float* data, const bool requires_grad = false) : _size(shape[0] * shape[1]), _shape(shape), _requires_grad(requires_grad) {
        cudaMalloc((float**)&_data, _size * sizeof(float));
        cudaMemcpy(_data, data, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
        } else {
            _grad = nullptr;
        }
    }

    explicit Tensor(const std::initializer_list<int> &shape, const bool requires_grad = false) : _shape(shape), _requires_grad(requires_grad) {
        _size = _shape[0] * _shape[1];
        cudaMalloc((float**)&_data, _size * sizeof(float));
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
        } else {
            _grad = nullptr;
        }
    }

    explicit Tensor(const std::initializer_list<int> &shape, const float* data, const bool requires_grad = false) : _shape(shape), _requires_grad(requires_grad) {
        _size = _shape[0] * _shape[1];
        cudaMalloc((float**)&_data, _size * sizeof(float));
        cudaMemcpy(_data, data, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
        } else {
            _grad = nullptr;
        }
    }

    template<typename Iterator>
    Tensor(const shape_type shape, Iterator first, Iterator last, const bool requires_grad = false) : _size(std::distance(first, last)), _shape(shape), _requires_grad(requires_grad) {
        if (shape[0]*shape[1] != _size) {
            throw std::invalid_argument("The size of the data does not match the shape.");
        }
        cudaMalloc((float**)&_data, _size * sizeof(float));
        cudaMemcpy(_data, first, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
        }
    }

    template<typename Iterator>
    Tensor(const std::initializer_list<int> &shape, Iterator first, Iterator last, const bool requires_grad = false) : _size(std::distance(first, last)), _shape(shape), _requires_grad(requires_grad) {
        if (_shape[0]*_shape[1] != _size) {
            throw std::invalid_argument("The size of the data does not match the shape.");
        }
        cudaMalloc((float**)&_data, _size * sizeof(float));
        cudaMemcpy(_data, first, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
        }
    }

    Tensor(const Tensor& other) : _size(other._size), _shape(other._shape), _requires_grad(other._requires_grad) {
        cudaMalloc((float**)&_data, _size * sizeof(float));
        cudaMemcpy(_data, other._data, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
            cudaMemcpy(_grad, other._grad, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            _grad = nullptr;
        }
    }

    Tensor(Tensor&& other) noexcept : _size(other._size), _shape(std::move(other._shape)), _requires_grad(other._requires_grad) {
        cudaMalloc((float**)&_data, _size * sizeof(float));
        cudaMalloc((float**)&_grad, _size * sizeof(float));
        cudaMemcpy(_data, other._data, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMemcpy(_grad, other._grad, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        other._data = nullptr;
        other._grad = nullptr;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            _size = other._size;
            _shape = other._shape;
            _requires_grad = other._requires_grad;
            cudaFree(_data);
            cudaMalloc((float**)&_data, _size * sizeof(float));
            cudaMemcpy(_data, other._data, _size * sizeof(float), cudaMemcpyDeviceToDevice);
            if (_requires_grad) {
                cudaFree(_grad);
                cudaMalloc((float**)&_grad, _size * sizeof(float));
                cudaMemcpy(_grad, other._grad, _size * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            _size = other._size;
            _shape = std::move(other._shape);
            cudaMemcpy(_data, other._data, _size * sizeof(float), cudaMemcpyDeviceToDevice);
            if (_requires_grad) {
                cudaMemcpy(_grad, other._grad, _size * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            other._data = nullptr;
            other._grad = nullptr;
        }
        return *this;
    }

    ~Tensor() {
        cudaFree(_data);
        if (_requires_grad) {
            cudaFree(_grad);
        }
    }

    bool requires_grad() const noexcept { return _requires_grad; }

    shape_type shape() const noexcept { return _shape; }

    size_type size() const noexcept { return _size; }

    void set_requires_grad(const bool requires_grad) noexcept {
        if (requires_grad && _grad == nullptr) {
            cudaMalloc((float**)_grad, _size * sizeof(float));
        }
        if (!requires_grad && _grad != nullptr) {
            cudaFree(_grad);
            _grad = nullptr;
        }
        _requires_grad = requires_grad;
    }

    void zero_grad() const noexcept {
        if (_requires_grad) {
            cudaMemset(_grad, 0, _size * sizeof(float));
        }
    }

    void print() const noexcept {
        std::ostream_iterator<float> output_iterator(std::cout, " ");
        float* data = (float*)malloc(_size * sizeof(float));
        cudaMemcpy(data, _data, _size * sizeof(float), cudaMemcpyDeviceToHost);
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

    void copy_data(const float* data, const shape_type &shape) {
        cudaFree(_data);
        if (_requires_grad) {
            cudaFree(_grad);
        }
        _size = shape[0] * shape[1];
        _shape = shape;
        cudaMalloc((float**)&_data, _size * sizeof(float));
        cudaMemcpy(_data, data, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
            cudaMemset(_grad, 0, _size * sizeof(float));
        }
    }

    void randomize(unsigned long long seed = 0) const {
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandGenerateUniform(gen, _data, _size);
    }

    void clear() const {
        cudaMemset(_data, 0, _size * sizeof(float));
    }

    void fill(const float value) const {
        cudaMemset(_data, value, _size * sizeof(float));
    }

    Tensor operator+(const Tensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor result(_shape, _requires_grad);
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        add_kernel<<<grid, block>>>(_data, other._data, result._data, _size);
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (_shape != other._shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor result(_shape, _requires_grad);
        dim3 block(256);
        dim3 grid((_size + block.x - 1) / block.x);
        sub_kernel<<<grid, block>>>(_data, other._data, result._data, _size);
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        if (_shape[1] != other._shape[0]) {
            throw std::invalid_argument("Matrix shapes do not match");
        }
        Tensor result({_shape[0], other._shape[1]}, _requires_grad);
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((_shape[0] + block.x - 1) / block.x, (_shape[1] + block.y - 1) / block.y);
        GEMM_kernel<<<grid, block>>>(_data, other._data, result._data, _shape[0], other._shape[1], _shape[1]);
        return result;
    }

    void reshape(const shape_type &shape) {
        float* temp = (float*)malloc(_size * sizeof(float));
        cudaMemcpy(temp, _data, _size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(_data);
        float* temp_grad = nullptr;
        if (_requires_grad) {
            temp_grad = (float*)malloc(_size * sizeof(float));
            cudaMemcpy(temp_grad, _grad, _size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(_grad);
        }
        size_type size = _size;
        _size = shape[0] * shape[1];
        _shape = shape;
        cudaMalloc((float**)&_data, _size * sizeof(float));
        cudaMemset(_data, 0, _size * sizeof(float));
        cudaMemcpy(_data, temp, size * sizeof(float), cudaMemcpyHostToDevice);
        free(temp);
        if (_requires_grad) {
            cudaMalloc((float**)&_grad, _size * sizeof(float));
            cudaMemset(_grad, 0, _size * sizeof(float));
            cudaMemcpy(_grad, temp_grad, size * sizeof(float), cudaMemcpyHostToDevice);
            free(temp_grad);
        }
    }

    void reshape(const std::initializer_list<int> &shape) {
        reshape(shape_type(shape));
    }

    void transpose() {
        float* temp;
        cudaMalloc((float**)&temp, _size * sizeof(float));
        cudaMemcpy(temp, _data, _size * sizeof(float), cudaMemcpyDeviceToDevice);
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((_shape[0] + block.x - 1) / block.x, (_shape[1] + block.y - 1) / block.y);
        Transpose_kernel<<<grid, block>>>(temp, _data, _shape[0], _shape[1]);
        reshape({_shape[1], _shape[0]});
        cudaFree(temp);
    }

    void set_data(const shape_type &position, const value_type value) const {
        value_type* data = (value_type*)malloc(_size * sizeof(value_type));
        cudaMemcpy(data, _data, _size * sizeof(value_type), cudaMemcpyDeviceToHost);
        data[position[0] * _shape[1] + position[1]] = value;
        cudaMemcpy(_data, data, _size * sizeof(value_type), cudaMemcpyHostToDevice);
        free(data);
    }

    void set_data(const std::initializer_list<int>& position, const value_type value) const {
        set_data(shape_type(position), value);
    }



private:
    size_type _size;
    shape_type _shape;
    float* _data;
    float* _grad;
    bool _requires_grad;
};

} // DL

#endif //TENSOR_CUH

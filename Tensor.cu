//
// Created by Administrator on 24-11-11.
//

#include "Tensor.cuh"

namespace DL {

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        float* data = (float*)malloc(tensor._size*sizeof(float));
        cudaMemcpy(data, tensor._data, tensor._size*sizeof(float), cudaMemcpyDeviceToHost);
        std::ostream_iterator<float> output_iterator(os, " ");
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
        float* data = (float*)malloc(tensor._size*sizeof(float));
        for (int i = 0; i < tensor._size; ++i) {
            is >> data[i];
        }
        cudaMemcpy(tensor._data, data, tensor._size*sizeof(float), cudaMemcpyHostToDevice);
        free(data);
        return is;
    }
} // DL
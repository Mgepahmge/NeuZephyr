#include "NeuZephyr/Dimension.cuh"
#include <iostream>

namespace nz::data {
    Dimension::Dimension(const size_t n, const size_t c, const size_t h, const size_t w) : n(n), c(c), h(h), w(w) {
        stride[0] = c * h * w;
        stride[1] = h * w;
        stride[2] = w;
        stride[3] = 1;
    }

    Dimension::Dimension() : Dimension(1, 1, 1, 1) {
    }

    Dimension::Dimension(const std::vector<size_t>& dims) : Dimension(dims[0], dims[1], dims[2], dims[3]) {
    }

    Dimension::Dimension(const Dimension& other) : Dimension(other.n, other.c, other.h, other.w) {
    }

    Dimension& Dimension::operator=(const Dimension& other) {
        if (this == &other) {
            return *this;
        }
        n = other.n;
        c = other.c;
        h = other.h;
        w = other.w;
        stride[0] = other.stride[0];
        stride[1] = other.stride[1];
        stride[2] = other.stride[2];
        stride[3] = other.stride[3];
        return *this;
    }

    size_t Dimension::size() const {
        return n * c * h * w;
    }

    size_t Dimension::getStride(const size_t i) const {
        if (!checkIndex(i)) {
            throw std::out_of_range("Index out of range");
        }
        return stride[i];
    }

    std::vector<size_t> Dimension::getDims() const {
        return {n, c, h, w};
    }

    size_t Dimension::N() const {
        return n;
    }

    size_t Dimension::C() const {
        return c;
    }

    size_t Dimension::H() const {
        return h;
    }

    size_t Dimension::W() const {
        return w;
    }

    size_t& Dimension::operator[](const size_t i) {
        switch (i) {
        case 0:
            return n;
        case 1:
            return c;
        case 2:
            return h;
        case 3:
            return w;
        default:
            throw std::out_of_range("Index out of range");
        }
    }

    const size_t& Dimension::operator[](const size_t i) const {
        switch (i) {
        case 0:
            return n;
        case 1:
            return c;
        case 2:
            return h;
        case 3:
            return w;
        default:
            throw std::out_of_range("Index out of range");
        }
    }

    bool Dimension::operator==(const Dimension& other) const {
        return n == other.n && c == other.c && h == other.h && w == other.w;
    }

    bool Dimension::isBroadcastCompatible(const Dimension& other) const {
        for (auto i = 0; i < 2; i++) {
            if (this->getDims()[i] != other.getDims()[i] && this->getDims()[i] != 1 && other.getDims()[i] != 1) {
                return false;
            }
        }
        return true;
    }

    bool Dimension::reshape(const Dimension& newShape) {
        if (newShape.size() != this->size()) {
            return false;
        }
        n = newShape.n;
        c = newShape.c;
        h = newShape.h;
        w = newShape.w;
        return true;
    }

    bool Dimension::operator!=(const Dimension& other) const {
        return !(*this == other);
    }

    Dimension Dimension::Broadcast(const Dimension& other) const {
        if (!isBroadcastCompatible(other)) {
            throw std::invalid_argument("Dimensions are not broadcast compatible");
        }
        Dimension result({0 ,0, 0, 0});
        for (auto i = 0; i < 4; i++) {
            result[i] = std::max(this->getDims()[i], other.getDims()[i]);
        }
        return result;
    }

    std::ostream& operator<<(std::ostream& os, const Dimension& dim) {
        os << dim.n << " " << dim.c << " " << dim.h << " " << dim.w;
        return os;
    }

    std::istream& operator>>(std::istream& is, Dimension& dim) {
        is >> dim.n >> dim.c >> dim.h >> dim.w;
        return is;
    }
}

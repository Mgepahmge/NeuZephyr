#ifndef DIMENSION_CUH
#define DIMENSION_CUH

#include <iostream>
#include <stdexcept>
#include <vector>

class Dimension {
public:
    friend std::ostream& operator<<(std::ostream& os, const Dimension& dim);

    friend std::istream& operator>>(std::istream& is, Dimension& dim);

    Dimension(size_t n, size_t c, size_t h, size_t w);

    Dimension();

    explicit Dimension(const std::vector<size_t>& dims);

    Dimension(const Dimension& other);

    Dimension& operator=(const Dimension& other);

    [[nodiscard]] size_t size() const;

    [[nodiscard]] size_t getStride(size_t i) const;

    [[nodiscard]] std::vector<size_t> getDims() const;

    [[nodiscard]] size_t N() const;

    [[nodiscard]] size_t C() const;

    [[nodiscard]] size_t H() const;

    [[nodiscard]] size_t W() const;

    [[nodiscard]] size_t operator[](size_t i) const;

    bool operator==(const Dimension& other) const;

    [[nodiscard]] bool isBroadcastCompatible(const Dimension& other) const;

private:
    size_t n;
    size_t c;
    size_t h;
    size_t w;
    size_t stride[4];

    static bool checkIndex(const size_t i) {
        return i < 4;
    }
};

#endif //DIMENSION_CUH

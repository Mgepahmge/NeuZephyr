#ifndef DIMENSION_CUH
#define DIMENSION_CUH

#include <iostream>
#include <stdexcept>
#include <vector>
#include "dl_export.cuh"

namespace nz::data {
    /**
     * @class Dimension
     * @brief Represents a multi - dimensional shape, typically used in deep learning for tensor dimensions.
     *
     * This class is designed to handle and manipulate multi - dimensional shapes commonly encountered in deep learning applications. It provides various methods for creating, comparing, reshaping, and broadcasting dimensions. The class uses four size_t variables (`n`, `c`, `h`, `w`) to store the dimensions and an array `stride` to store the corresponding strides.
     *
     * ### Type Definitions:
     * - There are no type definitions in this class.
     *
     * @details
     * ### Key Features:
     * - **Initialization**: Supports multiple ways of initializing a `Dimension` object, including direct specification of `n`, `c`, `h`, `w`, using a `std::vector<size_t>`, and copy construction.
     * - **Stream Operators**: Overloads the `<<` and `>>` operators for easy input and output of `Dimension` objects.
     * - **Accessors**: Provides methods to access individual dimensions (`N()`, `C()`, `H()`, `W()`), the number of elements (`size()`), strides (`getStride()`), and all dimensions as a `std::vector<size_t>` (`getDims()`).
     * - **Comparison**: Overloads the `==` and `!=` operators to compare two `Dimension` objects for equality.
     * - **Broadcast Compatibility**: Offers a method `isBroadcastCompatible()` to check if two `Dimension` objects can be broadcasted to each other, and a `Broadcast()` method to perform the actual broadcasting.
     * - **Reshaping**: Provides a `reshape()` method to change the shape of the `Dimension` object.
     *
     * ### Usage Example:
     * ```cpp
     * #include <iostream>
     * #include <vector>
     *
     * class DL_API Dimension {
     *     // Class definition as provided
     * };
     *
     * int main() {
     *     Dimension dim1(1, 3, 224, 224);
     *     Dimension dim2(std::vector<size_t>{1, 3, 224, 224});
     *     if (dim1 == dim2) {
     *         std::cout << "Dimensions are equal." << std::endl;
     *     }
     *     return 0;
     * }
     * ```
     *
     * @note
     * - Ensure that the indices used for accessing elements via `operator[]` are within the valid range (0 - 3). The private `checkIndex` method is used internally to enforce this.
     * - When using the `reshape` method, ensure that the new shape is valid and appropriate for the context.
     *
     * @author
     * Mgepahmge(https://github.com/Mgepahmge)
     *
     * @date
     * 2024/07/11
     */
    class DL_API Dimension {
    public:
        friend DL_API std::ostream& operator<<(std::ostream& os, const Dimension& dim);

        friend DL_API std::istream& operator>>(std::istream& is, Dimension& dim);

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

        [[nodiscard]] size_t& operator[](size_t i);

        bool operator==(const Dimension& other) const;

        [[nodiscard]] bool isBroadcastCompatible(const Dimension& other) const;

        bool reshape(const Dimension& newShape);

        bool operator!=(const Dimension& other) const;

        [[nodiscard]] Dimension Broadcast(const Dimension& other) const;

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
}


#endif //DIMENSION_CUH

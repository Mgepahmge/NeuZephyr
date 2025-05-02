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

        /**
         * @brief Constructs a Dimension object with specified dimensions and calculates the corresponding strides.
         *
         * @param n The batch size dimension. Memory flow: host-to-object, as the value is passed from the calling code to the object's member variable.
         * @param c The channel dimension. Memory flow: host-to-object, as the value is passed from the calling code to the object's member variable.
         * @param h The height dimension. Memory flow: host-to-object, as the value is passed from the calling code to the object's member variable.
         * @param w The width dimension. Memory flow: host-to-object, as the value is passed from the calling code to the object's member variable.
         *
         * @return None. This is a constructor, so it doesn't return a value.
         *
         * This constructor initializes a `Dimension` object with the provided batch size (`n`), channel count (`c`), height (`h`), and width (`w`). It then calculates the strides for each dimension based on these values. The stride for a dimension represents the number of elements to skip in the underlying data array to move to the next element along that dimension.
         *
         * **Memory Management Strategy**:
         * - This constructor does not allocate or free any dynamic memory. It simply initializes member variables of the `Dimension` object.
         *
         * **Exception Handling Mechanism**:
         * - There is no specific exception handling in this constructor. It assumes that the input values (`n`, `c`, `h`, `w`) are valid non - negative `size_t` values.
         *
         * **Relationship with Other Components**:
         * - The constructed `Dimension` object can be used by other parts of the system that rely on the concept of multi - dimensional data layout, such as tensor manipulation functions.
         *
         * @note
         * - Ensure that the input values (`n`, `c`, `h`, `w`) are non - negative `size_t` values, as negative values may lead to undefined behavior.
         * - The time complexity of this constructor is O(1) since it performs a fixed number of arithmetic operations regardless of the input values.
         *
         * @code
         * ```cpp
         * Dimension dim(2, 3, 4, 5);
         * ```
         * @endcode
         */
        Dimension(size_t n, size_t c, size_t h, size_t w);

        /**
         * @brief Constructs a Dimension object with default dimensions.
         *
         * This constructor initializes a Dimension object with default values for batch size (n = 1),
         * channel count (c = 1), height (h = 1), and width (w = 1). It achieves this by delegating
         * the initialization to the four - parameter constructor of the Dimension class.
         *
         * @param None. This is a default constructor, so it does not take any parameters.
         *
         * @return None. This is a constructor, so it does not return a value.
         *
         * **Memory Management Strategy**:
         * - This constructor does not allocate or free any dynamic memory. The memory management
         *   is handled by the four - parameter constructor it delegates to.
         *
         * **Exception Handling Mechanism**:
         * - Any exceptions that may occur during the initialization are handled by the four - parameter
         *   constructor. This default constructor does not have its own exception - handling logic.
         *
         * **Relationship with Other Components**:
         * - It provides a convenient way to create a Dimension object with default values, which can
         *   be used as a starting point in other parts of the system that rely on the Dimension class.
         *
         * @note
         * - The time complexity of this constructor is O(1) because it simply calls another constructor
         *   with fixed arguments.
         * - Ensure that the four - parameter constructor of the Dimension class is correctly implemented,
         *   as this constructor depends on it.
         *
         * @code
         * ```cpp
         * Dimension defaultDim;
         * ```
         * @endcode
         */
        Dimension();

        /**
         * @brief Constructs a Dimension object using a vector of dimensions.
         *
         * This constructor initializes a Dimension object by extracting the first four elements from the provided vector of `size_t` values. It then delegates the actual initialization to the four - parameter constructor of the Dimension class.
         *
         * @param dims A reference to a std::vector<size_t> containing the dimensions. Memory flow: host - to - object, as the values from the vector are used to initialize the object's member variables.
         *
         * @return None. This is a constructor, so it does not return a value.
         *
         * **Memory Management Strategy**:
         * - This constructor does not allocate or free any dynamic memory. It only accesses the elements of the input vector and delegates the memory management to the four - parameter constructor.
         *
         * **Exception Handling Mechanism**:
         * - If the input vector `dims` has less than four elements, accessing `dims[0]`, `dims[1]`, `dims[2]`, or `dims[3]` will result in undefined behavior. The four - parameter constructor may also throw exceptions if the input values are invalid. This constructor does not have its own exception - handling logic.
         *
         * **Relationship with Other Components**:
         * - It provides a convenient way to create a Dimension object when the dimensions are stored in a vector. Other parts of the system that generate or manipulate dimensions in vector form can use this constructor.
         *
         * @note
         * - Ensure that the input vector `dims` contains at least four elements to avoid undefined behavior.
         * - The time complexity of this constructor is O(1) because it performs a fixed number of operations regardless of the size of the input vector.
         *
         * @warning
         * - Accessing elements of the vector without checking its size can lead to undefined behavior.
         *
         * @code
         * ```cpp
         * std::vector<size_t> dims = {2, 3, 4, 5};
         * Dimension dim(dims);
         * ```
         * @endcode
         */
        explicit Dimension(const std::vector<size_t>& dims);

        /**
         * @brief Copy constructor for the Dimension class.
         *
         * This constructor creates a new Dimension object by copying the dimensions from an existing Dimension object. It delegates the actual initialization to the four - parameter constructor of the Dimension class.
         *
         * @param other A reference to an existing Dimension object from which the dimensions will be copied. Memory flow: object - to - object, as the values from the existing object are used to initialize the new object.
         *
         * @return None. This is a constructor, so it does not return a value.
         *
         * **Memory Management Strategy**:
         * - This constructor does not allocate or free any dynamic memory. It simply copies the member variables of the existing object and delegates the memory management to the four - parameter constructor.
         *
         * **Exception Handling Mechanism**:
         * - Any exceptions that may occur during the initialization are handled by the four - parameter constructor. This copy constructor does not have its own exception - handling logic.
         *
         * **Relationship with Other Components**:
         * - It provides a standard way to create a copy of a Dimension object, which can be useful in scenarios such as passing objects by value or creating backups.
         *
         * @note
         * - The time complexity of this constructor is O(1) because it performs a fixed number of operations regardless of the state of the input object.
         * - Ensure that the four - parameter constructor of the Dimension class is correctly implemented, as this constructor depends on it.
         *
         * @code
         * ```cpp
         * Dimension original(1, 2, 3, 4);
         * Dimension copy(original);
         * ```
         * @endcode
         */
        Dimension(const Dimension& other);

        /**
         * @brief Overloads the assignment operator for the Dimension class.
         *
         * This function assigns the values of an existing Dimension object to the current object. It first checks for self - assignment to avoid unnecessary operations. If the objects are different, it copies the dimensions and strides from the source object to the current object.
         *
         * @param other A reference to an existing Dimension object whose values will be assigned to the current object. Memory flow: object - to - object, as the values from the existing object are copied to the current object.
         *
         * @return A reference to the current Dimension object after the assignment.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It only copies the member variables of the source object to the current object.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions. It performs simple member - variable assignments, which are generally safe operations.
         *
         * **Relationship with Other Components**:
         * - It allows for the assignment of one Dimension object to another, which is useful in scenarios such as updating the state of an object or in algorithms that involve the manipulation of Dimension objects.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a fixed number of operations regardless of the state of the input object.
         * - Ensure that the member variables `n`, `c`, `h`, `w`, and `stride` in the Dimension class are correctly defined and accessible.
         *
         * @code
         * ```cpp
         * Dimension source(1, 2, 3, 4);
         * Dimension target;
         * target = source;
         * ```
         * @endcode
         */
        Dimension& operator=(const Dimension& other);

        /**
         * @brief Calculates the total number of elements in the Dimension object.
         *
         * This function computes the product of the four dimensions (`n`, `c`, `h`, `w`) of the Dimension object, which represents the total number of elements.
         *
         * @param None. This is a member function, so it operates on the current object.
         *
         * @return A `size_t` value representing the total number of elements in the Dimension object.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It only performs a simple arithmetic operation on the member variables of the object.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions. The multiplication operation is a basic arithmetic operation and is assumed to be safe within the range of `size_t`.
         *
         * **Relationship with Other Components**:
         * - The result of this function can be used in other parts of the program to allocate memory, iterate over elements, or perform other operations that depend on the total number of elements in the Dimension object.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a fixed number of arithmetic operations.
         * - Ensure that the member variables `n`, `c`, `h`, and `w` are non - negative to avoid unexpected results.
         *
         * @code
         * ```cpp
         * Dimension dim(2, 3, 4, 5);
         * size_t totalElements = dim.size();
         * ```
         * @endcode
         */
        [[nodiscard]] size_t size() const;

        /**
         * @brief Retrieves the stride value at a specified index within the Dimension object.
         *
         * This function checks if the given index is within the valid range using the `checkIndex` function. If the index is valid, it returns the corresponding stride value; otherwise, it throws an `std::out_of_range` exception.
         *
         * @param i The index of the stride value to retrieve. Memory flow: host-to-function, as the index value is passed from the calling code to the function.
         *
         * @return A `size_t` value representing the stride at the specified index.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It only accesses the existing `stride` array within the object.
         *
         * **Exception Handling Mechanism**:
         * - If the index `i` is out of range (i.e., `checkIndex(i)` returns `false`), the function throws an `std::out_of_range` exception with the message "Index out of range".
         *
         * **Relationship with Other Components**:
         * - This function depends on the `checkIndex` function to validate the index. The retrieved stride value can be used in other parts of the program for memory access calculations or other operations related to the data layout.
         *
         * @throws std::out_of_range If the provided index `i` is out of the valid range.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a constant number of operations (index check and array access).
         * - Ensure that the `checkIndex` function is correctly implemented to accurately validate the index.
         *
         * @warning
         * - Incorrectly passing an out-of-range index will result in an exception being thrown, which may cause the program to terminate if not properly handled.
         *
         * @code
         * ```cpp
         * Dimension dim;
         * try {
         *     size_t strideValue = dim.getStride(2);
         * } catch (const std::out_of_range& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] size_t getStride(size_t i) const;

        /**
         * @brief Retrieves the dimensions of the Dimension object as a std::vector.
         *
         * This function creates and returns a std::vector containing the four dimensions (n, c, h, w) of the Dimension object.
         *
         * @param None. This is a member function, so it operates on the current object.
         *
         * @return A std::vector<size_t> containing the dimensions (n, c, h, w) of the Dimension object. The memory for the vector is allocated on the heap, and ownership of the vector is transferred to the caller.
         *
         * **Memory Management Strategy**:
         * - The function allocates memory on the heap for the std::vector and its elements. The caller is responsible for managing the lifetime of the returned vector. When the vector goes out of scope, its destructor will automatically free the allocated memory.
         *
         * **Exception Handling Mechanism**:
         * - This function may throw a std::bad_alloc exception if there is not enough memory available to allocate the std::vector.
         *
         * **Relationship with Other Components**:
         * - The returned vector can be used in other parts of the program for further calculations or to pass the dimensions to other functions.
         *
         * @throws std::bad_alloc If there is not enough memory to allocate the std::vector.
         *
         * @note
         * - The time complexity of this function is O(1) because it creates a vector with a fixed number of elements (4 in this case).
         * - Ensure that the dimensions (n, c, h, w) are in the correct state before calling this function.
         *
         * @code
         * ```cpp
         * Dimension dim(1, 2, 3, 4);
         * std::vector<size_t> dimensions = dim.getDims();
         * for (size_t dim : dimensions) {
         *     std::cout << dim << " ";
         * }
         * std::cout << std::endl;
         * ```
         * @endcode
         */
        [[nodiscard]] std::vector<size_t> getDims() const;

        /**
         * @brief Retrieves the value of the 'n' dimension.
         *
         * This function is used to obtain the value of the 'n' dimension from the current object.
         *
         * @param None. This is a member function, so it operates on the current object.
         *
         * @return A `size_t` value representing the 'n' dimension. Memory flow: function-to-host, as the value is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It simply returns a value from the object's internal state.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances.
         *
         * **Relationship with Other Components**:
         * - The returned 'n' value can be used in other parts of the program for calculations related to the data layout or for passing to other functions that require this dimension information.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a simple value retrieval.
         * - Ensure that the 'n' dimension has been properly initialized before calling this function.
         *
         * @code
         * ```cpp
         * Dimension dim;
         * size_t nValue = dim.N();
         * std::cout << "n value: " << nValue << std::endl;
         * ```
         * @endcode
         */
        [[nodiscard]] size_t N() const;

        /**
         * @brief Retrieves the value of the 'c' dimension.
         *
         * This function is used to obtain the value of the 'c' dimension from the current object.
         *
         * @param None. This is a member function, so it operates on the current object.
         *
         * @return A `size_t` value representing the 'c' dimension. Memory flow: function-to-host, as the value is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It simply returns a value from the object's internal state.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances.
         *
         * **Relationship with Other Components**:
         * - The returned 'c' value can be used in other parts of the program for calculations related to the data layout or for passing to other functions that require this dimension information.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a simple value retrieval.
         * - Ensure that the 'c' dimension has been properly initialized before calling this function.
         *
         * @code
         * ```cpp
         * Dimension dim;
         * size_t cValue = dim.C();
         * std::cout << "c value: " << cValue << std::endl;
         * ```
         * @endcode
         */
        [[nodiscard]] size_t C() const;

        /**
         * @brief Retrieves the value of the 'h' dimension.
         *
         * This function is used to obtain the value of the 'h' dimension from the current object.
         *
         * @param None. This is a member function, so it operates on the current object.
         *
         * @return A `size_t` value representing the 'h' dimension. Memory flow: function-to-host, as the value is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It simply returns a value from the object's internal state.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances.
         *
         * **Relationship with Other Components**:
         * - The returned 'h' value can be used in other parts of the program for calculations related to the data layout or for passing to other functions that require this dimension information.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a simple value retrieval.
         * - Ensure that the 'h' dimension has been properly initialized before calling this function.
         *
         * @code
         * ```cpp
         * Dimension dim;
         * size_t hValue = dim.H();
         * std::cout << "h value: " << hValue << std::endl;
         * ```
         * @endcode
         */
        [[nodiscard]] size_t H() const;

        /**
         * @brief Retrieves the value of the 'w' dimension.
         *
         * This function is used to obtain the value of the 'w' dimension from the current object.
         *
         * @param None. This is a member function, so it operates on the current object.
         *
         * @return A `size_t` value representing the 'w' dimension. Memory flow: function-to-host, as the value is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It simply returns a value from the object's internal state.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances.
         *
         * **Relationship with Other Components**:
         * - The returned 'w' value can be used in other parts of the program for calculations related to the data layout or for passing to other functions that require this dimension information.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a simple value retrieval.
         * - Ensure that the 'w' dimension has been properly initialized before calling this function.
         *
         * @code
         * ```cpp
         * Dimension dim;
         * size_t wValue = dim.W();
         * std::cout << "w value: " << wValue << std::endl;
         * ```
         * @endcode
         */
        [[nodiscard]] size_t W() const;

        /**
         * @brief Overloads the subscript operator to access the dimensions of the Dimension object.
         *
         * This non - const version of the subscript operator allows for both accessing and modifying the dimensions of the Dimension object.
         *
         * @param i A `size_t` value representing the index of the dimension to access. Memory flow: host - to - function, as the index is passed from the calling code to the function.
         *
         * @return A reference to a `size_t` representing the requested dimension. Memory flow: function - to - host, as a reference to the internal dimension is returned to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It simply returns a reference to an existing member variable.
         *
         * **Exception Handling Mechanism**:
         * - If the provided index `i` is not in the range `[0, 3]`, a `std::out_of_range` exception is thrown.
         *
         * **Relationship with Other Components**:
         * - This operator can be used in expressions where direct access or modification of the dimensions is required, such as in loops for iterating over dimensions or in calculations involving specific dimensions.
         *
         * @throws std::out_of_range If the index `i` is not in the range `[0, 3]`.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a simple switch statement to return the appropriate dimension.
         * - Ensure that the index `i` is in the valid range `[0, 3]` to avoid exceptions.
         *
         * @code
         * ```cpp
         * Dimension dim;
         * dim[0] = 10; // Modify the 'n' dimension
         * std::cout << "Modified n dimension: " << dim[0] << std::endl;
         * ```
         * @endcode
         */
        [[nodiscard]] size_t& operator[](size_t i);

        /**
         * @brief Overloads the subscript operator to access the dimensions of the const Dimension object.
         *
         * This const version of the subscript operator allows for read - only access to the dimensions of the Dimension object.
         *
         * @param i A `size_t` value representing the index of the dimension to access. Memory flow: host - to - function, as the index is passed from the calling code to the function.
         *
         * @return A const reference to a `size_t` representing the requested dimension. Memory flow: function - to - host, as a const reference to the internal dimension is returned to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It simply returns a const reference to an existing member variable.
         *
         * **Exception Handling Mechanism**:
         * - If the provided index `i` is not in the range `[0, 3]`, a `std::out_of_range` exception is thrown.
         *
         * **Relationship with Other Components**:
         * - This operator can be used in expressions where read - only access to the dimensions of a const Dimension object is required, such as in functions that take a const reference to a Dimension object.
         *
         * @throws std::out_of_range If the index `i` is not in the range `[0, 3]`.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a simple switch statement to return the appropriate dimension.
         * - Ensure that the index `i` is in the valid range `[0, 3]` to avoid exceptions.
         *
         * @code
         * ```cpp
         * const Dimension dim;
         * size_t value = dim[1]; // Access the 'c' dimension
         * std::cout << "Value of c dimension: " << value << std::endl;
         * ```
         * @endcode
         */
        [[nodiscard]] const size_t& operator[](size_t i) const;

        /**
         * @brief Compares two `Dimension` objects for equality.
         *
         * This function checks if all corresponding dimensions (n, c, h, w) of the current `Dimension` object
         * are equal to those of another `Dimension` object.
         *
         * @param other A constant reference to another `Dimension` object to compare with. Memory flow: host-to-function, as the object is passed from the calling code to the function.
         *
         * @return A boolean value indicating whether the two `Dimension` objects are equal. Memory flow: function-to-host, as the result is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It only performs comparison operations on the member variables of the objects.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances.
         *
         * **Relationship with Other Components**:
         * - This operator can be used in conditional statements, loops, or other parts of the program where equality comparison of `Dimension` objects is required, such as in data validation or sorting algorithms.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a fixed number of comparison operations.
         * - Ensure that both `Dimension` objects are properly initialized before calling this function.
         *
         * @code
         * ```cpp
         * Dimension dim1;
         * Dimension dim2;
         * bool areEqual = dim1 == dim2;
         * std::cout << "Are dimensions equal? " << (areEqual ? "Yes" : "No") << std::endl;
         * ```
         * @endcode
         */
        bool operator==(const Dimension& other) const;

        /**
         * @brief Checks if the current Dimension object is broadcast compatible with another Dimension object.
         *
         * This function determines whether the dimensions of two objects can be broadcast together according to the broadcasting rules.
         * For each of the first two dimensions, it checks if the dimensions are equal or if one of them is 1. If all checks pass for the first two dimensions, the objects are considered broadcast compatible.
         *
         * @param other A constant reference to another Dimension object to compare with. Memory flow: host-to-function, as the object is passed from the calling code to the function.
         *
         * @return A boolean value indicating whether the two Dimension objects are broadcast compatible. Memory flow: function-to-host, as the result is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It only accesses the existing dimensions of the objects.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances. It assumes that the `getDims()` method of both objects returns valid dimension arrays.
         *
         * **Relationship with Other Components**:
         * - This function is useful in operations where broadcasting of dimensions is required, such as in element-wise operations between tensors with different dimensions. It can be used to validate if the dimensions of two tensors can be broadcast before performing the actual operation.
         *
         * @note
         * - The time complexity of this function is O(1) because it iterates over a fixed number (2) of dimensions.
         * - Ensure that both Dimension objects have valid dimensions before calling this function.
         *
         * @code
         * ```cpp
         * Dimension dim1;
         * Dimension dim2;
         * bool isCompatible = dim1.isBroadcastCompatible(dim2);
         * std::cout << "Are dimensions broadcast compatible? " << (isCompatible ? "Yes" : "No") << std::endl;
         * ```
         * @endcode
         */
        [[nodiscard]] bool isBroadcastCompatible(const Dimension& other) const;

        /**
         * @brief Attempts to reshape the current `Dimension` object to a new shape.
         *
         * This function checks if the size of the new shape is equal to the size of the current `Dimension` object.
         * If they are equal, it updates the current object's dimensions (n, c, h, w) to match the new shape and returns `true`.
         * Otherwise, it does not modify the current object and returns `false`.
         *
         * @param newShape A constant reference to a `Dimension` object representing the new shape. Memory flow: host-to-function, as the object is passed from the calling code to the function.
         *
         * @return A boolean value indicating whether the reshape operation was successful. Memory flow: function-to-host, as the result is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It only modifies the member variables of the current `Dimension` object.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances. It simply returns `false` if the reshape is not possible.
         *
         * **Relationship with Other Components**:
         * - This function can be used in scenarios where the shape of a data structure needs to be changed, such as in tensor reshaping operations. It ensures that the total number of elements remains the same after the reshape.
         *
         * @note
         * - The time complexity of this function is O(1) because it performs a fixed number of operations.
         * - Ensure that the `size()` method of both `Dimension` objects returns valid sizes before calling this function.
         *
         * @code
         * ```cpp
         * Dimension currentDim;
         * Dimension newDim;
         * bool success = currentDim.reshape(newDim);
         * if (success) {
         *     std::cout << "Reshape operation was successful." << std::endl;
         * } else {
         *     std::cout << "Reshape operation failed." << std::endl;
         * }
         * ```
         * @endcode
         */
        bool reshape(const Dimension& newShape);

        /**
         * @brief Overloads the '!=' operator to compare two Dimension objects for inequality.
         *
         * This function determines whether the current Dimension object is not equal to another Dimension object.
         * It achieves this by negating the result of the equality comparison ('==') between the two objects.
         *
         * @param other A constant reference to another Dimension object to compare with. Memory flow: host-to-function, as the object is passed from the calling code to the function.
         *
         * @return A boolean value indicating whether the two Dimension objects are not equal. Memory flow: function-to-host, as the result is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function does not allocate or free any dynamic memory. It only performs a comparison operation on the existing objects.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances. It assumes that the '==' operator for Dimension objects is properly defined and does not throw exceptions.
         *
         * **Relationship with Other Components**:
         * - This operator overload is useful in scenarios where inequality checks are required, such as in conditional statements or sorting algorithms. It provides a convenient way to compare two Dimension objects.
         *
         * @note
         * - The time complexity of this function depends on the implementation of the '==' operator for Dimension objects. If the '==' operator has a time complexity of O(k), then this function also has a time complexity of O(k).
         * - Ensure that the '==' operator for Dimension objects is correctly implemented before using this '!=' operator.
         *
         * @code
         * ```cpp
         * Dimension dim1;
         * Dimension dim2;
         * if (dim1 != dim2) {
         *     std::cout << "The two dimensions are not equal." << std::endl;
         * } else {
         *     std::cout << "The two dimensions are equal." << std::endl;
         * }
         * ```
         * @endcode
         */
        bool operator!=(const Dimension& other) const;

        /**
         * @brief Performs broadcasting between two `Dimension` objects and returns the resulting `Dimension`.
         *
         * This function checks if the current `Dimension` object and the provided `other` `Dimension` object are broadcast compatible.
         * If they are, it creates a new `Dimension` object where each dimension is the maximum of the corresponding dimensions of the two input `Dimension` objects.
         * Otherwise, it throws an `std::invalid_argument` exception.
         *
         * @param other A constant reference to another `Dimension` object to perform broadcasting with. Memory flow: host-to-function, as the object is passed from the calling code to the function.
         *
         * @return A new `Dimension` object representing the result of the broadcasting operation. Memory flow: function-to-host, as the result is returned from the function to the calling code.
         *
         * **Memory Management Strategy**:
         * - This function creates a new `Dimension` object (`result`) on the stack. The object is automatically destroyed when it goes out of scope.
         *
         * **Exception Handling Mechanism**:
         * - If the dimensions are not broadcast compatible, this function throws an `std::invalid_argument` exception.
         *
         * **Relationship with Other Components**:
         * - This function relies on the `isBroadcastCompatible` method to check the compatibility of the dimensions. It is useful in scenarios where element-wise operations between tensors of different shapes are required, such as in deep learning frameworks.
         *
         * @throws std::invalid_argument If the dimensions are not broadcast compatible.
         *
         * @note
         * - The time complexity of this function is O(4) = O(1) because it iterates over a fixed number (4) of dimensions.
         * - Ensure that the `isBroadcastCompatible` method is correctly implemented and that the `getDims` method returns the appropriate dimension values.
         *
         * @code
         * ```cpp
         * Dimension dim1;
         * Dimension dim2;
         * try {
         *     Dimension result = dim1.Broadcast(dim2);
         *     // Use the result
         * } catch (const std::invalid_argument& e) {
         *     std::cerr << e.what() << std::endl;
         * }
         * ```
         * @endcode
         */
        [[nodiscard]] Dimension Broadcast(const Dimension& other) const;

        /**
         * @brief Updates the stride values of the Dimension object.
         *
         * This function calculates and assigns new stride values based on the current values of c, h, and w in the Dimension object.
         * The stride values are used to determine the memory layout and access pattern for multi - dimensional data.
         *
         * @param None
         *
         * @return None
         *
         * **Memory Management Strategy**:
         * - This function only modifies the existing member variable `stride` of the `Dimension` object. It does not allocate or free any dynamic memory.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw any exceptions under normal circumstances. It assumes that the member variables `c`, `h`, and `w` are properly initialized.
         *
         * **Relationship with Other Components**:
         * - The updated stride values are likely used in other parts of the program for accessing multi - dimensional data efficiently. For example, in tensor operations or data access routines.
         *
         * @note
         * - The time complexity of this function is O(1) as it involves a fixed number of arithmetic operations.
         * - Ensure that the member variables `c`, `h`, and `w` are correctly initialized before calling this function.
         *
         * @code
         * ```cpp
         * Dimension dim;
         * dim.updateStride();
         * ```
         * @endcode
         */
        void updateStride();

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
